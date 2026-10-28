"""Build a self-contained optimization dataset from recorded radio traces.

This module deliberately has no simulator, scheduler, or cluster dependency.
Each optimization is stored as a separate checked shard so an interrupted
build can resume without relabeling results from a different configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .dataloader.episodes import episode_shapes, load_episode
from .optimization.milp import HandoverOptimizerMILP
from .optimization.optim_config import OptimConfig
from .plotting.rate_connected_time_vs_lambda_plot import (
    plot_rate_connectivity_tradeoff,
)
from .plotting.rate_outage_pareto_frontiers_plot import plot_pareto_fronts
from .reference.reference import RRCReferenceSimulation
from .reference.rrc_config import RRCConfig


_ENVIRONMENT_FIELDS = {
    "base_dir", "log_dir", "path_to_config", "simulation_id", "log_level",
    "save_dir", "dataset_path", "show_plots", "save_plots",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _scientific_config(config: OptimConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json", exclude=_ENVIRONMENT_FIELDS)
    payload["rrc"] = config.rrc.model_dump(mode="json", exclude=_ENVIRONMENT_FIELDS)
    return payload


def _a3_parameters(config: RRCConfig) -> dict[str, float]:
    event = config.ho_event_config.get("a3")
    if not isinstance(event, dict):
        raise ValueError("Each A3 profile requires rrc.ho_event_config.a3")
    missing = {"ttt_ms", "hys", "offset"} - set(event)
    if missing:
        raise ValueError(f"A3 profile is missing parameters: {sorted(missing)}")
    return {
        "a3_ttt_ms": float(event["ttt_ms"]),
        "a3_hys_db": float(event["hys"]),
        "a3_off_dbm": float(event["offset"]),
    }


def _validate_episode(episode: Any, rrc: RRCConfig) -> tuple[int, int]:
    rsrp = np.asarray(episode.rsrp)
    sinr = np.asarray(episode.sinr)
    if rsrp.shape != sinr.shape or sinr.ndim not in (2, 3) or sinr.size == 0:
        raise ValueError("Episode RSRP and SINR must be nonempty matching 2-D or 3-D arrays")
    if not np.isfinite(rsrp).all() or not np.isfinite(sinr).all():
        raise ValueError("Episode RSRP and SINR must contain only finite values")
    stored = episode.config.get("t_res_ms")
    if stored is None and isinstance(episode.config.get("rrc"), dict):
        stored = episode.config["rrc"].get("t_res_ms")
    if stored is not None and int(stored) != rrc.t_res_ms:
        raise ValueError(
            f"RRC time resolution {rrc.t_res_ms} ms differs from stored {stored} ms"
        )
    return int(sinr.shape[0]), int(sinr.shape[1]) if sinr.ndim == 3 else 1


def _step_frame(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> pd.DataFrame:
    n_steps = int(np.asarray(arrays["pcell"]).size)
    columns: dict[str, Any] = {key: [value] * n_steps for key, value in metadata.items()}
    columns["time_step"] = np.arange(n_steps, dtype=np.int32)
    for name, value in arrays.items():
        data = np.asarray(value)
        if data.ndim == 1 and data.shape[0] == n_steps:
            columns[name] = data
        elif data.ndim == 2 and data.shape[1] == n_steps:
            columns[name] = [data[:, step].tolist() for step in range(n_steps)]
    return pd.DataFrame(columns)


def _run_milp(episode: Any, ue_idx: int, config: OptimConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    optimizer = HandoverOptimizerMILP(config)
    optimizer.load_data(episode, ue_idx)
    optimizer.setup_model()
    optimizer.apply_warm_start_if_configured()
    optimizer.solve()
    if optimizer.model.SolCount == 0:
        raise RuntimeError(f"MILP found no solution (status={optimizer.model.Status})")
    if optimizer.model.Status != 2:
        raise RuntimeError(f"MILP did not prove optimality (status={optimizer.model.Status})")
    metrics = optimizer.get_aggregated_result_metrics().copy()
    metrics.pop("simulation_id", None)
    metrics["obj_value"] = metrics["ho_objective_value"]
    return metrics, _step_frame(optimizer.extract_results(), {})


def _run_a3(episode: Any, rrc: RRCConfig, backend: str) -> list[dict[str, Any]]:
    if backend == "standalone":
        reference = RRCReferenceSimulation(rrc, episode)
        reference.run()
        rows = list(reference.get_per_ue_result_metrics().values())
    elif backend == "simulator":
        from .integrations.simulator import run_a3_reference

        rows = run_a3_reference(episode, rrc)
    else:
        raise ValueError(f"Unknown A3 backend: {backend}")
    for row in rows:
        row.pop("simulation_id", None)
        row["ue_id"] = row.pop("ue_idx", row.get("ue_id"))
    return rows


def _check_shard(
    path: Path, expected_fingerprint: str, *, step_path: Path | None = None
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Existing result has a different configuration: {path}")
    if step_path is not None:
        if not step_path.is_file():
            return False
        if payload.get("steps_sha256") != _sha256(step_path):
            raise ValueError(f"Existing step result failed integrity check: {step_path}")
    return True


def build_dataset(
    network_data: Mapping[str, str | Path],
    config: OptimConfig,
    lambdas: Sequence[float],
    output_root: str | Path,
    *,
    a3_profiles: Sequence[RRCConfig] | None = None,
    a3_backend: str = "standalone",
) -> Path:
    """Build or resume a dataset from stored HDF5 network measurements.

    A complete marker is written only after every expected result and figure
    exists. Existing partial shards are reused only for the identical input
    hashes and scientific configuration.
    """
    if not network_data:
        raise ValueError("At least one network_data file is required")
    if not lambdas or any(not np.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("lambdas must be finite, non-negative, and nonempty")
    if len(set(lambdas)) != len(lambdas):
        raise ValueError("lambdas must be unique")
    if a3_backend not in {"standalone", "simulator"}:
        raise ValueError("a3_backend must be standalone or simulator")
    profiles = list(a3_profiles) if a3_profiles is not None else [config.rrc]
    if not profiles:
        raise ValueError("At least one A3 profile is required")
    for profile in profiles:
        _a3_parameters(profile)

    root = Path(output_root).expanduser().resolve()
    if root.is_dir() and any(root.iterdir()) and not (root / "dataset.yaml").is_file():
        raise ValueError(f"Output directory is not an existing dataset build: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "release_complete.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {root}")

    source_paths = {split: Path(path).expanduser().resolve() for split, path in network_data.items()}
    for split, source in source_paths.items():
        if not split or "/" in split or "\\" in split or split in {".", ".."}:
            raise ValueError(f"Invalid split name: {split!r}")
        if not source.is_file() or source.suffix.lower() != ".h5":
            raise FileNotFoundError(f"Network HDF5 file not found: {source}")
        if root in source.parents:
            raise ValueError("Source network_data must be outside output_root")

    source_hashes = {split: _sha256(path) for split, path in source_paths.items()}
    spec = {
        "schema_version": 1,
        "network_data": {split: {"file": f"network_data/{split}.h5", "sha256": source_hashes[split]}
                         for split in sorted(source_paths)},
        "optim_config": _scientific_config(config),
        "lambdas": [float(value) for value in lambdas],
        "a3_backend": a3_backend,
        "a3_profiles": [profile.model_dump(mode="json", exclude=_ENVIRONMENT_FIELDS)
                        for profile in profiles],
    }
    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")
    ).hexdigest()
    spec_path = root / "dataset.yaml"
    if spec_path.exists():
        stored_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if stored_spec != spec:
            raise ValueError("Existing dataset.yaml describes different inputs or configuration")
    else:
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=True), encoding="utf-8")

    for split, source in source_paths.items():
        destination = root / "network_data" / f"{split}.h5"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256(destination) != source_hashes[split]:
                raise ValueError(f"Copied network_data has changed: {destination}")
        else:
            temporary = destination.with_suffix(".h5.tmp")
            shutil.copy2(source, temporary)
            if _sha256(temporary) != source_hashes[split]:
                raise ValueError(f"Network data copy failed integrity check: {source}")
            temporary.replace(destination)

        shapes = episode_shapes(destination)
        if not shapes:
            raise ValueError(f"No episodes in {destination}")
        for ep_idx, (ep_name, _, n_ue, _) in enumerate(shapes):
            episode = load_episode(destination, ep_idx)
            _, validated_ues = _validate_episode(episode, config.rrc)
            if validated_ues != n_ue:
                raise ValueError(f"UE count mismatch in {ep_name}")
            for lambda_idx, lambda_r in enumerate(lambdas):
                row_config = config.model_copy(update={"lambda_r": float(lambda_r)})
                for ue_idx in range(n_ue):
                    stem = f"{ep_name}_ue_{ue_idx:04d}_lambda_{lambda_idx:04d}"
                    row_file = root / "optim_results" / split / "rows" / f"{stem}.json"
                    step_file = root / "optim_results" / split / "steps" / f"{stem}.parquet"
                    if _check_shard(row_file, fingerprint, step_path=step_file):
                        continue
                    metrics, steps = _run_milp(episode, ue_idx, row_config)
                    metadata = {"split": split, "episode_idx": ep_idx, "episode_name": ep_name,
                                "ue_id": ue_idx, "lambda": float(lambda_r),
                                "model_profile": config.model_profile}
                    _write_parquet(step_file, steps.assign(**metadata))
                    _write_json(row_file, {"fingerprint": fingerprint,
                                           "steps_sha256": _sha256(step_file),
                                           "metrics": {**metrics, **metadata}})
            for profile_idx, profile in enumerate(profiles):
                stem = f"{ep_name}_a3_{profile_idx:04d}"
                row_file = root / "reference_results" / split / "rows" / f"{stem}.json"
                if _check_shard(row_file, fingerprint):
                    continue
                rows = _run_a3(episode, profile, a3_backend)
                for row in rows:
                    row.update({"split": split, "episode_idx": ep_idx,
                                "episode_name": ep_name, "a3_backend": a3_backend,
                                **_a3_parameters(profile)})
                _write_json(row_file, {"fingerprint": fingerprint, "metrics": rows})

        optim_rows = [json.loads(path.read_text(encoding="utf-8"))["metrics"]
                      for path in sorted((root / "optim_results" / split / "rows").glob("*.json"))]
        reference_rows = [row
                          for path in sorted((root / "reference_results" / split / "rows").glob("*.json"))
                          for row in json.loads(path.read_text(encoding="utf-8"))["metrics"]]
        expected_optim = sum(n_ue for _, _, n_ue, _ in shapes) * len(lambdas)
        expected_reference = sum(n_ue for _, _, n_ue, _ in shapes) * len(profiles)
        if len(optim_rows) != expected_optim or len(reference_rows) != expected_reference:
            raise RuntimeError(f"Incomplete results for split {split}")
        optim_path = root / "optim_results" / split / "optim_result_metrics.parquet"
        reference_path = root / "reference_results" / split / "reference_result_metrics.parquet"
        _write_parquet(optim_path, pd.DataFrame(optim_rows))
        _write_parquet(reference_path, pd.DataFrame(reference_rows))
        plot_dir = root / "plots" / split
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_pareto_fronts(str(optim_path), str(reference_path), str(plot_dir),
                           auto_limits=True)
        plot_rate_connectivity_tradeoff(str(optim_path), str(reference_path),
                                         str(plot_dir), auto_limits=True)

    artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
         "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"release_complete.json", "dataset_manifest.json"}
    ]
    _write_json(root / "dataset_manifest.json", {"fingerprint": fingerprint,
                                                  "artifacts": artifacts})
    _write_json(root / "release_complete.json", {"fingerprint": fingerprint,
                                                   "artifact_count": len(artifacts)})
    return root


def validate_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Verify a completed dataset's source and output files against its manifest."""
    root = Path(dataset_root).expanduser().resolve()
    spec_path = root / "dataset.yaml"
    manifest_path = root / "dataset_manifest.json"
    marker_path = root / "release_complete.json"
    for path in (spec_path, manifest_path, marker_path):
        if not path.is_file():
            raise ValueError(f"Incomplete dataset: missing {path.name}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    if manifest.get("fingerprint") != fingerprint or marker.get("fingerprint") != fingerprint:
        raise ValueError("Dataset configuration fingerprint does not match")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or marker.get("artifact_count") != len(artifacts):
        raise ValueError("Dataset artifact count does not match")
    expected: set[str] = set()
    for artifact in artifacts:
        relative = Path(artifact["path"])
        path = (root / relative).resolve()
        if path == root or root not in path.parents or relative.as_posix() in expected:
            raise ValueError(f"Invalid dataset artifact path: {relative}")
        expected.add(relative.as_posix())
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"Dataset artifact failed integrity check: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual - {"dataset_manifest.json", "release_complete.json"} != expected:
        raise ValueError("Dataset contains unlisted or missing artifacts")
    for source in spec["network_data"].values():
        if not any(item["path"] == source["file"] and item["sha256"] == source["sha256"]
                   for item in artifacts):
            raise ValueError(f"Network source missing from manifest: {source['file']}")
    return {"fingerprint": fingerprint, "artifact_count": len(artifacts)}
