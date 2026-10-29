"""Build and publish datasets from stored radio trajectories."""

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
import mobile_network_sim
from mobile_network_sim.rrc import RRCConfig

from .. import __version__
from ..optimization.optim_config import OptimConfig
from .a3_selection import DEFAULT_SPEED_BINS, build_selection_table
from .a3_protocol import A3SearchProtocol
from .recorded_steps import (
    _BuildRows, _a3_profile, _merge_parquet_parts, _parameters,
    _run_recorded_episodes, _run_refined_a3_search, _run_selected_a3,
)
from .reporting import publish_reports
from .storage import ENVIRONMENT_FIELDS as _ENVIRONMENT_FIELDS
from .storage import scientific_config as _scientific_config, sha256 as _sha256
from .storage import write_json as _write_json, write_parquet as _write_parquet


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_root(
    network_data: Mapping[str, str | Path], config: OptimConfig,
    lambdas: Sequence[float], profiles: Sequence[RRCConfig], output_root: str | Path,
    *, selection_split: str, speed_bins: Sequence[tuple[float, float]],
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    if not network_data or selection_split not in network_data:
        raise ValueError("An explicit A3 selection split in network_data is required")
    if not lambdas or len(set(lambdas)) != len(lambdas) or any(
        not np.isfinite(value) or value < 0 for value in lambdas
    ):
        raise ValueError("Lambda values must be distinct, finite and non-negative")
    if not profiles:
        raise ValueError("At least one explicit A3 search profile is required")
    parameters = [_parameters(profile) for profile in profiles]
    if len(set(parameters)) != len(parameters):
        raise ValueError("A3 search profiles must be unique")
    for profile in profiles:
        if profile.t_res_ms != config.rrc.t_res_ms:
            raise ValueError("A3 and MILP sample intervals must match")
        if _parameters(profile)[0] % profile.t_res_ms:
            raise ValueError("A3 TTT must be a multiple of t_res_ms")
    if not speed_bins or any(low < 0 or high <= low for low, high in speed_bins):
        raise ValueError("Speed bins must be nonempty increasing intervals")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Dataset output already exists: {root}")
    sources = {split: Path(path).expanduser().resolve() for split, path in network_data.items()}
    for split, path in sources.items():
        if not split or split in (".", "..") or "/" in split or "\\" in split:
            raise ValueError(f"Invalid split name: {split!r}")
        if not path.is_file() or path.suffix.lower() != ".h5":
            raise FileNotFoundError(path)
        if root in path.parents:
            raise ValueError("Source network_data must be outside output_root")
    spec = {
        "schema_version": 3,
        "release": {"release_id": root.name, "lambda_values": [float(x) for x in lambdas],
                    "capacity_sinr_source": "raw",
                    "splits": [{"name": split} for split in sorted(sources)]},
        "milp_variants": {
            "unrestricted": {"layout": "legacy", "root": "optim_results"},
            "binary_rank1": {
                "layout": "episode_shards", "root": "optim_results_binary_rank1",
                "initial_access": "strongest_rsrp",
                "handover_target_mode": "rank1_neighbor",
                "handover_target_signal": "raw_rsrp",
                "published_episodes": {}, "complete": False,
            },
        },
        "network_data": {split: {"file": f"network_data/{split}.h5", "sha256": _sha256(path)}
                         for split, path in sorted(sources.items())},
        "artifacts": {
            "network_data": "network_data/<split>.h5",
            "milp_metrics": "optim_results/<split>/optim_result_metrics.parquet",
            "milp_trajectories": "optim_results/<split>/optim_results.parquet",
            "no_ho_metrics": "reference_results/no_ho/<split>/metrics.parquet",
            "a3_metrics": "reference_results/a3/<level>/<split>/metrics.parquet",
            "a3_trajectories": "reference_results/a3/<level>/<split>/trajectories.parquet",
            "rrc_events": "events/<split>/rrc_events.parquet",
            "handover_attempts": "events/<split>/handover_attempts.parquet",
            "outage_decomposition": "events/<split>/outage_decomposition.parquet",
        },
        "optim_config": _scientific_config(config),
        "rrc_engine": "mobile-network-sim",
        "software": {
            "ho_optim_milp": __version__,
            "mobile_network_sim": mobile_network_sim.__version__,
        },
        "a3_selection_split": selection_split,
        "a3_profiles": [profile.model_dump(mode="json", exclude=_ENVIRONMENT_FIELDS)
                        for profile in profiles],
        "speed_bins_kph": [list(pair) for pair in speed_bins],
    }
    root.mkdir(parents=True)
    (root / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=True), encoding="utf-8")
    for split, source in sources.items():
        target = root / "network_data" / f"{split}.h5"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha256(target) != spec["network_data"][split]["sha256"]:
            raise ValueError(f"Network data copy failed integrity check: {split}")
    return root, sources, spec


def _write_results(
    root: Path, sources: Mapping[str, Path], config: OptimConfig,
    rows: _BuildRows,
) -> None:
    """Merge result tables and generate figures."""
    parts = rows.parts
    optim_rows = rows.optim_rows
    no_ho_rows = rows.no_ho_rows
    selected_rows = rows.selected_rows

    for split in sorted(sources):
        _write_parquet(
            root / "optim_results" / split / "optim_result_metrics.parquet",
            pd.DataFrame([row for row in optim_rows if row["split"] == split]),
        )
        _write_parquet(
            root / "reference_results" / "no_ho" / split / "metrics.parquet",
            pd.DataFrame([row for row in no_ho_rows if row["split"] == split]),
        )
        for level in ("global", "speed_aware", "split_oracle"):
            _write_parquet(
                root / "reference_results" / "a3" / level / split / "metrics.parquet",
                pd.DataFrame([row for row in selected_rows
                              if row["split"] == split and row["selection_level"] == level]),
            )
            key = (f"reference_results/a3/{level}", split)
            _merge_parquet_parts(
                parts[key], root / key[0] / split / "trajectories.parquet"
            )
        _merge_parquet_parts(
            parts[("optim_results", split)],
            root / "optim_results" / split / "optim_results.parquet",
        )
        for name in ("rrc_events", "handover_attempts", "outage_decomposition"):
            _merge_parquet_parts(
                parts[(name, split)], root / "events" / split / f"{name}.parquet"
            )

    selected_frame = pd.DataFrame(selected_rows)
    operating = (
        selected_frame.groupby(
            ["selection_level", "split", "lambda", "selected_on", "deployable",
             "ttt_ms", "hys_db", "offset_db"], dropna=False, as_index=False
        )[["mean_capacity", "rel_connected_time", "simulated_time_s",
           "num_ho", "num_pp", "num_rlf", "num_hof"]].mean()
    )
    operating["outage"] = 1.0 - operating["rel_connected_time"]
    operating["score"] = operating["mean_capacity"] - operating["lambda"] * operating["outage"]
    _write_parquet(root / "reference_results" / "a3" / "operating_points.parquet", operating)
    _write_parquet(root / "reference_results" / "a3_operating_points.parquet", operating)

    all_metrics = pd.DataFrame([*optim_rows, *no_ho_rows, *selected_rows])
    all_attempts = pd.concat(
        [pd.read_parquet(path) for (name, _), paths in parts.items()
         if name == "handover_attempts" for path in paths], ignore_index=True
    )
    all_outage = pd.concat(
        [pd.read_parquet(path) for (name, _), paths in parts.items()
         if name == "outage_decomposition" for path in paths], ignore_index=True
    )
    publish_reports(
        metrics=all_metrics, attempts=all_attempts, outage=all_outage,
        output_root=root, q_in_db=config.rrc.q_in_db, q_out_db=config.rrc.q_out_db,
        report_splits=tuple(sorted(sources)),
    )


def _finalize_release(root: Path, spec: dict[str, Any], rows: _BuildRows) -> Path:
    """Write the manifest after all dataset artifacts are complete."""
    work = root / "_work"
    release_id = root.name
    binary_published = rows.binary_published
    spec["milp_variants"]["binary_rank1"]["published_episodes"] = binary_published
    spec["milp_variants"]["binary_rank1"]["complete"] = True
    (root / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=True), encoding="utf-8")
    if work.resolve().parent != root:
        raise RuntimeError("Internal work directory escaped output root")
    shutil.rmtree(work)
    artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
         "sha256": _sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
        and path.name not in ("dataset_manifest.json", "release_complete.json")
    ]
    fingerprint = _fingerprint(spec)
    _write_json(root / "dataset_manifest.json", {
        "schema_version": 3, "fingerprint": fingerprint, "artifacts": artifacts,
    })
    _write_json(root / "release_complete.json", {
        "schema_version": 3, "release_id": release_id,
        "dataset_manifest": "dataset_manifest.json",
        "dataset_manifest_sha256": _sha256(root / "dataset_manifest.json"),
        "fingerprint": fingerprint, "artifact_count": len(artifacts), "passed": True,
        "milp_recomputed": True,
    })
    return root


def build_recorded_dataset(
    network_data: Mapping[str, str | Path],
    config: OptimConfig,
    lambdas: Sequence[float],
    output_root: str | Path,
    *,
    a3_profiles: Sequence[RRCConfig],
    selection_split: str,
    a3_protocol: A3SearchProtocol | None = None,
    speed_bins: Sequence[tuple[float, float]] = DEFAULT_SPEED_BINS,
) -> Path:
    """Build a schema-3 dataset from stored HDF5 radio measurements."""
    profiles = list(a3_profiles)
    template = None
    if a3_protocol is not None:
        if len(profiles) != 1:
            raise ValueError("A3 search requires exactly one template profile")
        template = profiles[0]
        profiles = [_a3_profile(template, parameters)
                    for parameters in a3_protocol.coarse_grid()]
    root, sources, spec = _prepare_root(
        network_data, config, lambdas, profiles, output_root,
        selection_split=selection_split, speed_bins=speed_bins,
    )
    rows = _BuildRows(binary_published={split: [] for split in sources})
    if a3_protocol is not None:
        spec["a3_search"] = {**a3_protocol.as_dict(),
                             "fingerprint": a3_protocol.fingerprint}
    _run_recorded_episodes(
        root, sources, config, lambdas, profiles, rows,
        search_stage="coarse" if a3_protocol else "explicit",
    )
    profile_lookup = {_parameters(profile): profile for profile in profiles}
    if a3_protocol is not None:
        assert template is not None
        grids = a3_protocol.refinement_grids(
            rows.search_rows, lambda_values=lambdas,
            selection_split=selection_split,
        )
        parameter_lambdas: dict[tuple[float, float, float], list[float]] = {}
        for lambda_r, parameters in grids.items():
            for parameter in parameters:
                parameter_lambdas.setdefault(parameter, []).append(lambda_r)
        profile_lookup.update(_run_refined_a3_search(
            root, sources, template, parameter_lambdas, rows,
        ))
        spec["a3_profiles"] = [
            profile.model_dump(mode="json", exclude=_ENVIRONMENT_FIELDS)
            for profile in profile_lookup.values()
        ]

    selections = build_selection_table(
        rows.search_rows, lambda_values=lambdas, selection_split=selection_split,
        evaluation_splits=sorted(sources), bins=speed_bins,
    )
    selection_frame = pd.DataFrame(selections)
    _write_parquet(root / "reference_results" / "a3" / "search_metrics.parquet",
                   pd.DataFrame(rows.search_rows))
    _write_parquet(root / "reference_results" / "a3" / "selections.parquet",
                   selection_frame)

    _run_selected_a3(
        root, sources, lambdas, selection_frame,
        profile_lookup,
        speed_bins, rows,
    )
    _write_results(root, sources, config, rows)
    return _finalize_release(root, spec, rows)
