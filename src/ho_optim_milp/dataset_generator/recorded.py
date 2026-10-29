"""Build schema-3 results from stored radio trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..dataloader.episodes import episode_shapes, load_episode
from ..dataset import EPISODE_MARKER_SCHEMA
from ..optimization.optim_config import OptimConfig
from ..reference.rrc_config import RRCConfig
from .a3_selection import DEFAULT_SPEED_BINS, build_selection_table, speed_bin_label
from .events import derive_handover_attempts, derive_outage_decomposition, derive_rrc_events
from .recorded_results import optimize_recorded_ue, reference_recorded_episode
from .reporting import publish_reports
from .storage import (
    ENVIRONMENT_FIELDS as _ENVIRONMENT_FIELDS,
    scientific_config as _scientific_config,
    sha256 as _sha256,
    validate_episode as _validate_episode,
    write_json as _write_json,
    write_parquet as _write_parquet,
)


def _parameters(config: RRCConfig) -> tuple[float, float, float]:
    event = config.ho_event_config.get("a3")
    if not isinstance(event, dict) or not {"ttt_ms", "hys", "offset"} <= set(event):
        raise ValueError("Every A3 profile needs ttt_ms, hys, and offset")
    return float(event["ttt_ms"]), float(event["hys"]), float(event["offset"])


def _speed_kph(episode: Any, ue_id: int) -> float:
    config = episode.config
    ues = config.get("ues")
    if ues is None and isinstance(config.get("radio"), dict):
        ues = config["radio"].get("ues")
    if isinstance(ues, list):
        if not 0 <= ue_id < len(ues):
            raise ValueError(f"UE speed metadata missing for UE {ue_id}")
        ue = ues[ue_id]
    elif isinstance(ues, dict):
        ue = ues
    else:
        raise ValueError("A3 speed-aware selection requires per-UE speed metadata")
    for name in ("gauss_markov_mean_speed_kph", "speed_kph", "speed_kmh"):
        if name in ue and ue[name] is not None:
            value = float(ue[name])
            if np.isfinite(value) and value >= 0:
                return value
    raise ValueError(f"No finite speed metadata for UE {ue_id}")


def _identity(
    *, release_id: str, split: str, method: str, episode_idx: int,
    ue_id: int, lambda_r: float, selection_level: str | None = None,
    selected_on: str | None = None, deployable: bool | None = None,
    bin_label: str | None = None, parameters: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    ttt, hys, offset = parameters if parameters is not None else (None, None, None)
    return {
        "release_id": release_id, "split": split, "method": method,
        "selection_level": selection_level, "lambda": float(lambda_r),
        "episode_idx": episode_idx, "ue_id": ue_id,
        "trajectory_id": f"{method}:{split}:{selection_level}:{lambda_r:g}:{episode_idx}:{ue_id}",
        "selected_on": selected_on, "evaluated_on": split,
        "deployable": deployable, "bin_label": bin_label,
        "ttt_ms": ttt, "hys_db": hys, "offset_db": offset,
    }


def _annotate(frame: pd.DataFrame, identity: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for name, value in identity.items():
        result[name] = value
    return result


def _milp_event_fields(frame: pd.DataFrame, config: RRCConfig) -> pd.DataFrame:
    """Recover idealized MILP handover phases from completion samples."""
    result = frame.copy()
    n_steps = len(result)
    for name in ("i_prep_start", "i_prep_expire", "i_exec_start", "i_exec_expire",
                 "i_ho_success", "i_rlf_no_ho"):
        result[name] = np.zeros(n_steps, dtype=np.int8)
    result["target_cell"] = pd.array([None] * n_steps, dtype="Int64")
    result["i_rlf_no_ho"] = result["i_rlf_start"].astype(np.int8)
    prep = config.t_ho_prep_ms_simulated // config.t_res_ms
    execution = config.t_ho_exec_ms_simulated // config.t_res_ms
    for finish in np.flatnonzero(result["i_ho_exec_end"].to_numpy(dtype=bool)):
        start = int(finish) - prep - execution
        if start < 0:
            raise ValueError(f"MILP handover completion at step {finish} is left-censored")
        prep_end = start + prep
        target = int(result["pcell"].iloc[finish])
        result.loc[start:finish, "target_cell"] = target
        result.loc[start, "i_prep_start"] = 1
        result.loc[prep_end, "i_prep_expire"] = 1
        result.loc[prep_end, "i_exec_start"] = 1
        result.loc[finish, "i_exec_expire"] = 1
        result.loc[finish, "i_ho_success"] = 1
    return result


def _derived_tables(
    frame: pd.DataFrame, config: RRCConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = derive_rrc_events(frame, t_res_ms=config.t_res_ms)
    attempts = derive_handover_attempts(frame, t_res_ms=config.t_res_ms)
    outage = derive_outage_decomposition(frame, attempts, t_res_ms=config.t_res_ms)
    return events, attempts, outage


def _merge_parquet_parts(parts: Sequence[Path], destination: Path) -> None:
    """Merge one trajectory at a time using a unified Arrow schema."""
    if not parts:
        raise ValueError(f"No result parts for {destination}")
    schemas = [pq.read_schema(path) for path in parts]
    schema = pa.unify_schemas(schemas)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with pq.ParquetWriter(temporary, schema) as writer:
        for path in parts:
            table = pq.read_table(path)
            for field in schema:
                if field.name not in table.column_names:
                    table = table.append_column(field.name, pa.nulls(len(table), type=field.type))
            writer.write_table(table.select(schema.names).cast(schema, safe=False))
    temporary.replace(destination)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_root(
    network_data: Mapping[str, str | Path], config: OptimConfig,
    lambdas: Sequence[float], profiles: Sequence[RRCConfig], output_root: str | Path,
    *, backend: str, selection_split: str, speed_bins: Sequence[tuple[float, float]],
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    if backend not in ("standalone", "simulator"):
        raise ValueError("A3 backend must be standalone or simulator")
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
        "a3_backend": backend,
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


def _emit_trajectory(
    frame: pd.DataFrame, config: RRCConfig, work: Path,
    parts: dict[tuple[str, str], list[Path]], *, split: str, key: str,
    trajectory_kind: str | None,
) -> None:
    if trajectory_kind is not None:
        path = work / trajectory_kind / split / f"{key}.parquet"
        _write_parquet(path, frame)
        parts.setdefault((trajectory_kind, split), []).append(path)
    events, attempts, outage = _derived_tables(frame, config)
    for name, table in (("rrc_events", events), ("handover_attempts", attempts),
                        ("outage_decomposition", outage)):
        path = work / "events" / split / name / f"{key}.parquet"
        _write_parquet(path, table)
        parts.setdefault((name, split), []).append(path)


def build_recorded_dataset(
    network_data: Mapping[str, str | Path],
    config: OptimConfig,
    lambdas: Sequence[float],
    output_root: str | Path,
    *,
    a3_profiles: Sequence[RRCConfig],
    selection_split: str,
    a3_backend: str = "standalone",
    speed_bins: Sequence[tuple[float, float]] = DEFAULT_SPEED_BINS,
) -> Path:
    """Create a schema-3 dataset from stored HDF5 radio measurements.

    A3 selection is restricted to the declared selection split. The split
    oracle is marked non-deployable. Every MILP solve uses a stored, bounded
    trajectory; this function never generates radio data.
    """
    profiles = list(a3_profiles)
    root, sources, spec = _prepare_root(
        network_data, config, lambdas, profiles, output_root,
        backend=a3_backend, selection_split=selection_split, speed_bins=speed_bins,
    )
    work = root / "_work"
    release_id = root.name
    parts: dict[tuple[str, str], list[Path]] = {}
    optim_rows: list[dict[str, Any]] = []
    binary_published: dict[str, list[int]] = {split: [] for split in sources}
    no_ho_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    profile_lookup = {_parameters(profile): profile for profile in profiles}

    for split, source in sorted(sources.items()):
        shapes = episode_shapes(source)
        if not shapes:
            raise ValueError(f"No network episodes in {source}")
        for episode_idx, (_, n_steps, n_ue, _) in enumerate(shapes):
            episode = load_episode(source, episode_idx)
            actual_steps, actual_ues = _validate_episode(episode, config.rrc)
            if (actual_steps, actual_ues) != (n_steps, n_ue):
                raise ValueError(f"Network episode shape changed: {episode.ep_name}")
            speeds = [_speed_kph(episode, ue_id) for ue_id in range(n_ue)]
            no_ho = reference_recorded_episode(
                episode, config.rrc, lambda_r=0.0, backend=a3_backend,
                trigger_handovers=False,
            )
            for ue_id, (base_metrics, base_frame) in enumerate(no_ho):
                for lambda_idx, lambda_r in enumerate(lambdas):
                    identity = _identity(
                        release_id=release_id, split=split, method="no_ho",
                        episode_idx=episode_idx, ue_id=ue_id, lambda_r=lambda_r,
                    )
                    metrics = {**base_metrics, **identity,
                               "mean_speed_kph": speeds[ue_id]}
                    metrics["ho_objective_value"] = (
                        metrics["mean_capacity"] - lambda_r *
                        (1.0 - metrics["rel_connected_time"])
                    )
                    no_ho_rows.append(metrics)
                    _emit_trajectory(
                        _annotate(base_frame, identity), config.rrc, work, parts,
                        split=split, key=f"no_ho_{episode_idx}_{ue_id}_{lambda_idx}",
                        trajectory_kind=None,
                    )

            binary_rows: list[dict[str, Any]] = []
            binary_parts: list[Path] = []
            for lambda_idx, lambda_r in enumerate(lambdas):
                row_config = config.model_copy(update={
                    "lambda_r": float(lambda_r), "initial_access": "free",
                    "handover_target_mode": "free",
                })
                binary_config = config.model_copy(update={
                    "lambda_r": float(lambda_r),
                    "initial_access": "strongest_rsrp",
                    "handover_target_mode": "rank1_neighbor",
                })
                for ue_id in range(n_ue):
                    metrics, frame = optimize_recorded_ue(episode, ue_id, row_config)
                    identity = _identity(
                        release_id=release_id, split=split, method="milp",
                        episode_idx=episode_idx, ue_id=ue_id, lambda_r=lambda_r,
                    )
                    optim_rows.append({**metrics, **identity,
                                       "mean_speed_kph": speeds[ue_id],
                                       "solver_status": "OPTIMAL"})
                    enriched = _milp_event_fields(_annotate(frame, identity), config.rrc)
                    _emit_trajectory(
                        enriched, config.rrc, work, parts, split=split,
                        key=f"milp_{episode_idx}_{ue_id}_{lambda_idx}",
                        trajectory_kind="optim_results",
                    )
                    binary_metrics, binary_frame = optimize_recorded_ue(
                        episode, ue_id, binary_config
                    )
                    binary_rows.append({
                        **binary_metrics, **identity, "milp_variant": "binary_rank1",
                        "mean_speed_kph": speeds[ue_id], "solver_status": "OPTIMAL",
                    })
                    binary_part = (work / "binary_rank1" / split
                                   / f"ep_{episode_idx:04d}_{lambda_idx}_{ue_id}.parquet")
                    _write_parquet(
                        binary_part, _milp_event_fields(binary_frame, config.rrc)
                    )
                    binary_parts.append(binary_part)

            binary_directory = (root / "optim_results_binary_rank1" / split
                                / "episodes" / f"ep_{episode_idx:04d}")
            binary_metrics_path = binary_directory / "metrics.parquet"
            binary_steps_path = binary_directory / "trajectories.parquet"
            _write_parquet(binary_metrics_path, pd.DataFrame(binary_rows))
            _merge_parquet_parts(binary_parts, binary_steps_path)
            binary_parquet = pq.ParquetFile(binary_steps_path)
            try:
                row_groups = binary_parquet.num_row_groups
            finally:
                binary_parquet.close()
            if row_groups != len(binary_rows):
                raise ValueError("Binary result row groups do not match metric rows")
            _write_json(binary_directory / "complete.json", {
                "schema": EPISODE_MARKER_SCHEMA, "split": split,
                "episode_idx": episode_idx, "rows": len(binary_rows),
                "steps": n_steps * len(binary_rows),
                "row_groups": row_groups,
                "files": {
                    "metrics.parquet": _sha256(binary_metrics_path),
                    "trajectories.parquet": _sha256(binary_steps_path),
                },
            })
            binary_published[split].append(episode_idx)

            for profile_idx, profile in enumerate(profiles):
                parameter_values = _parameters(profile)
                results = reference_recorded_episode(
                    episode, profile, lambda_r=0.0, backend=a3_backend
                )
                for ue_id, (base_metrics, _) in enumerate(results):
                    for lambda_r in lambdas:
                        identity = _identity(
                            release_id=release_id, split=split, method="a3",
                            episode_idx=episode_idx, ue_id=ue_id, lambda_r=lambda_r,
                            parameters=parameter_values,
                        )
                        metrics = {**base_metrics, **identity,
                                   "mean_speed_kph": speeds[ue_id],
                                   "a3_backend": a3_backend, "profile_idx": profile_idx}
                        metrics["ho_objective_value"] = (
                            metrics["mean_capacity"] - lambda_r *
                            (1.0 - metrics["rel_connected_time"])
                        )
                        search_rows.append(metrics)

    selections = build_selection_table(
        search_rows, lambda_values=lambdas, selection_split=selection_split,
        evaluation_splits=sorted(sources), bins=speed_bins,
    )
    selection_frame = pd.DataFrame(selections)
    _write_parquet(root / "reference_results" / "a3" / "search_metrics.parquet",
                   pd.DataFrame(search_rows))
    _write_parquet(root / "reference_results" / "a3" / "selections.parquet",
                   selection_frame)

    selected_rows: list[dict[str, Any]] = []
    for split, source in sorted(sources.items()):
        for episode_idx, (_, _, n_ue, _) in enumerate(episode_shapes(source)):
            episode = load_episode(source, episode_idx)
            speeds = [_speed_kph(episode, ue_id) for ue_id in range(n_ue)]
            cached: dict[tuple[float, float, float], list[tuple[dict[str, Any], pd.DataFrame]]] = {}
            for lambda_idx, lambda_r in enumerate(lambdas):
                for level in ("global", "speed_aware", "split_oracle"):
                    for ue_id in range(n_ue):
                        candidates = selection_frame[
                            (selection_frame["level"] == level)
                            & (selection_frame["lambda"] == lambda_r)
                        ]
                        if level == "speed_aware":
                            label = speed_bin_label(speeds[ue_id], speed_bins)
                            if label is None and speeds[ue_id] > speed_bins[-1][1]:
                                low, high = speed_bins[-1]
                                label = f"[{low:g},{high:g}]"
                            if label is None:
                                raise ValueError(f"UE speed outside declared bins: {speeds[ue_id]}")
                            candidates = candidates[candidates["bin_label"] == label]
                        elif level == "split_oracle":
                            candidates = candidates[candidates["evaluated_on"] == split]
                        if len(candidates) != 1:
                            raise ValueError(
                                f"Expected one {level} A3 selection for {split}, lambda={lambda_r}"
                            )
                        selection = candidates.iloc[0]
                        parameters = (float(selection.ttt_ms), float(selection.hys_db),
                                      float(selection.offset_db))
                        if parameters not in cached:
                            cached[parameters] = reference_recorded_episode(
                                episode, profile_lookup[parameters], lambda_r=0.0,
                                backend=a3_backend,
                            )
                        base_metrics, base_frame = cached[parameters][ue_id]
                        identity = _identity(
                            release_id=release_id, split=split, method="a3",
                            episode_idx=episode_idx, ue_id=ue_id, lambda_r=lambda_r,
                            selection_level=level, selected_on=str(selection.selected_on),
                            deployable=bool(selection.deployable),
                            bin_label=(str(selection.bin_label)
                                       if pd.notna(selection.bin_label) else None),
                            parameters=parameters,
                        )
                        metrics = {**base_metrics, **identity,
                                   "mean_speed_kph": speeds[ue_id],
                                   "a3_backend": a3_backend}
                        metrics["ho_objective_value"] = (
                            metrics["mean_capacity"] - lambda_r *
                            (1.0 - metrics["rel_connected_time"])
                        )
                        selected_rows.append(metrics)
                        _emit_trajectory(
                            _annotate(base_frame, identity), profile_lookup[parameters],
                            work, parts, split=split,
                            key=f"a3_{level}_{episode_idx}_{ue_id}_{lambda_idx}",
                            trajectory_kind=f"reference_results/a3/{level}",
                        )

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
