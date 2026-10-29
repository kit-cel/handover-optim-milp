"""Per-episode MILP and A3 result generation from recorded traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mobile_network_sim.rrc import RRCConfig

from ..dataloader.episodes import episode_shapes, load_episode
from ..dataset import EPISODE_MARKER_SCHEMA
from ..optimization.optim_config import OptimConfig
from .a3_selection import speed_bin_label
from .events import derive_handover_attempts, derive_outage_decomposition, derive_rrc_events
from .recorded_results import optimize_recorded_ue, reference_recorded_episode
from .rrc_replay import replay_stored_rrc_trajectory
from .storage import sha256 as _sha256, validate_episode as _validate_episode
from .storage import write_json as _write_json, write_parquet as _write_parquet


def _parameters(config: RRCConfig) -> tuple[float, float, float]:
    event = config.ho_event_config.get("a3")
    if not isinstance(event, dict) or not {"ttt_ms", "hys", "offset"} <= set(event):
        raise ValueError("Every A3 profile needs ttt_ms, hys, and offset")
    return float(event["ttt_ms"]), float(event["hys"]), float(event["offset"])


def _a3_profile(
    template: RRCConfig, parameters: tuple[float, float, float]
) -> RRCConfig:
    """Return a validated RRC profile with the requested A3 parameters."""
    payload = template.model_dump(mode="python")
    events = dict(payload["ho_event_config"])
    a3 = dict(events.get("a3", {}))
    a3.update(zip(("ttt_ms", "hys", "offset"), parameters))
    events["a3"] = a3
    payload["ho_event_config"] = events
    return RRCConfig.model_validate(payload)


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


def _milp_event_fields(
    frame: pd.DataFrame, config: RRCConfig, episode: Any, ue_id: int
) -> pd.DataFrame:
    """Add validated simulator RRC events to a solved MILP trajectory."""
    replay = replay_stored_rrc_trajectory(
        config, episode, ue_id=ue_id, stored_steps=frame, method="milp"
    )
    result = frame.copy()
    for name in replay:
        if name not in result:
            result[name] = replay[name].to_numpy()
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


@dataclass
class _BuildRows:
    parts: dict[tuple[str, str], list[Path]] = field(default_factory=dict)
    optim_rows: list[dict[str, Any]] = field(default_factory=list)
    binary_published: dict[str, list[int]] = field(default_factory=dict)
    no_ho_rows: list[dict[str, Any]] = field(default_factory=list)
    search_rows: list[dict[str, Any]] = field(default_factory=list)
    selected_rows: list[dict[str, Any]] = field(default_factory=list)


def _run_refined_a3_search(
    root: Path, sources: Mapping[str, Path], template: RRCConfig,
    parameter_lambdas: Mapping[tuple[float, float, float], Sequence[float]],
    rows: _BuildRows,
) -> dict[tuple[float, float, float], RRCConfig]:
    """Score refined A3 profiles on stored episodes without resolving MILPs."""
    profiles = {parameters: _a3_profile(template, parameters)
                for parameters in parameter_lambdas}
    for split, source in sorted(sources.items()):
        for episode_idx, (_, _, n_ue, _) in enumerate(episode_shapes(source)):
            episode = load_episode(source, episode_idx)
            speeds = [_speed_kph(episode, ue_id) for ue_id in range(n_ue)]
            for parameters, profile in profiles.items():
                results = reference_recorded_episode(episode, profile, lambda_r=0.0)
                for lambda_r in parameter_lambdas[parameters]:
                    for ue_id, (base_metrics, _) in enumerate(results):
                        identity = _identity(
                            release_id=root.name, split=split, method="a3",
                            episode_idx=episode_idx, ue_id=ue_id,
                            lambda_r=lambda_r, parameters=parameters,
                        )
                        metrics = {
                            **base_metrics,
                            **identity,
                            "mean_speed_kph": speeds[ue_id],
                            "search_stage": "fine",
                        }
                        metrics["ho_objective_value"] = (
                            metrics["mean_capacity"] - lambda_r *
                            (1.0 - metrics["rel_connected_time"])
                        )
                        rows.search_rows.append(metrics)
    return profiles


def _run_recorded_episodes(
    root: Path, sources: Mapping[str, Path], config: OptimConfig,
    lambdas: Sequence[float], profiles: Sequence[RRCConfig],
    rows: _BuildRows, *, search_stage: str = "explicit",
) -> None:
    """Compute MILP, no-handover, and A3 search rows for each episode."""
    work = root / "_work"
    release_id = root.name
    parts = rows.parts
    optim_rows = rows.optim_rows
    binary_published = rows.binary_published
    no_ho_rows = rows.no_ho_rows
    search_rows = rows.search_rows

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
                episode, config.rrc, lambda_r=0.0,
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
                    enriched = _milp_event_fields(
                        _annotate(frame, identity), config.rrc, episode, ue_id
                    )
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
                        binary_part, _milp_event_fields(
                            binary_frame, config.rrc, episode, ue_id
                        )
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
                    episode, profile, lambda_r=0.0
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
                                   "profile_idx": profile_idx,
                                   "search_stage": search_stage}
                        metrics["ho_objective_value"] = (
                            metrics["mean_capacity"] - lambda_r *
                            (1.0 - metrics["rel_connected_time"])
                        )
                        search_rows.append(metrics)


def _run_selected_a3(
    root: Path, sources: Mapping[str, Path], lambdas: Sequence[float],
    selection_frame: pd.DataFrame, profile_lookup: Mapping[tuple[float, float, float], RRCConfig],
    speed_bins: Sequence[tuple[float, float]], rows: _BuildRows,
) -> None:
    """Replay the selected A3 profiles and store their trajectories."""
    work = root / "_work"
    release_id = root.name
    parts = rows.parts
    selected_rows = rows.selected_rows

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
                                   }
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
