"""Derive RRC events, handover attempts, and disjoint outage.

All functions consume canonical per-step traces.  They do not evaluate A3 or
advance timers; state-machine semantics remain owned by ``RRCStepUE``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..common.schema import (
    STEP_PER_CELL_L1_QIN_SINR_PREFIX,
    STEP_PER_CELL_L1_QOUT_SINR_PREFIX,
    STEP_PER_CELL_L3_RSRP_PREFIX,
    STEP_PER_CELL_L3_SINR_PREFIX,
    STEP_PER_CELL_QIN_PREFIX,
    STEP_PER_CELL_QOUT_PREFIX,
    STEP_PER_CELL_SINR_PREFIX,
)

IDENTITY_COLUMNS = (
    "release_id",
    "split",
    "method",
    "selection_level",
    "lambda",
    "episode_idx",
    "ue_id",
    "trajectory_id",
    "selected_on",
    "evaluated_on",
    "deployable",
    "bin_label",
    "ttt_ms",
    "hys_db",
    "offset_db",
)

EVENT_FLAGS = {
    "ttt_start": "ttt_start",
    "ttt_cancel": "ttt_cancel",
    "ttt_expire": "ttt_expire",
    "prep_start": "prep_start",
    "prep_cancel": "prep_cancel",
    "prep_expire": "prep_end",
    "exec_start": "exec_start",
    "exec_cancel": "exec_cancel",
    "exec_expire": "exec_end",
    "ho_success": "ho_success",
    "ho_failure": "ho_failure",
    "t310_start": "t310_start",
    "t310_cancel": "t310_cancel",
    "t310_expire": "t310_expire",
    "rlf_start": "rlf_start",
    "rlf_expire": "rlf_end",
    "pp_detected": "pp_detected",
    "rlf_no_ho": "rlf_no_ho",
    "hof_prep": "hof_prep",
    "hof_cmd": "hof_cmd",
    "hof_exec": "hof_exec",
}

ATTEMPT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "attempt_id",
    "ttt_start_step",
    "ttt_expire_step",
    "prep_start_step",
    "prep_end_step",
    "exec_start_step",
    "exec_end_step",
    "source_cell",
    "target_cell",
    "outcome",
    "failure_cause",
    "censored_phase",
    "prep_duration_samples",
    "exec_duration_samples",
    "total_duration_samples",
    "prep_duration_s",
    "exec_duration_s",
    "total_duration_s",
)

_SIGNALS = {
    "raw_sinr_db": STEP_PER_CELL_SINR_PREFIX,
    "l3_rsrp_dbm": STEP_PER_CELL_L3_RSRP_PREFIX,
    "l3_sinr_db": STEP_PER_CELL_L3_SINR_PREFIX,
    "l1_qin_sinr_db": STEP_PER_CELL_L1_QIN_SINR_PREFIX,
    "l1_qout_sinr_db": STEP_PER_CELL_L1_QOUT_SINR_PREFIX,
    "qin_indicator": STEP_PER_CELL_QIN_PREFIX,
    "qout_indicator": STEP_PER_CELL_QOUT_PREFIX,
}


def _stable_derived_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent all-null trajectory slices from producing Arrow null fields."""
    result = frame.copy()
    string_columns = (
        "release_id",
        "split",
        "method",
        "selection_level",
        "trajectory_id",
        "selected_on",
        "evaluated_on",
        "bin_label",
        "event_type",
        "attempt_id",
        "outcome",
        "failure_cause",
        "censored_phase",
    )
    int_columns = (
        "episode_idx",
        "ue_id",
        "time_step",
        "time_ms",
        "pcell",
        "source_cell",
        "target_cell",
        "state",
        "ttt_start_step",
        "ttt_expire_step",
        "prep_start_step",
        "prep_end_step",
        "exec_start_step",
        "exec_end_step",
        "prep_duration_samples",
        "exec_duration_samples",
        "total_duration_samples",
    )
    bool_columns = ("deployable", "up_connected", "rlm_evaluated")
    float_columns = (
        "lambda",
        "ttt_ms",
        "hys_db",
        "offset_db",
        "prep_duration_s",
        "exec_duration_s",
        "total_duration_s",
    )
    for column in string_columns:
        if column in result:
            result[column] = pd.array(result[column], dtype="string")
    for column in int_columns:
        if column in result:
            result[column] = pd.array(result[column], dtype="Int64")
    for column in bool_columns:
        if column in result:
            result[column] = pd.array(result[column], dtype="boolean")
    for column in float_columns:
        if column in result:
            result[column] = pd.array(result[column], dtype="Float64")
    for column in result:
        if any(column.endswith(f"_{name}") for name in _SIGNALS):
            result[column] = pd.array(result[column], dtype="Float64")
        elif column.endswith("_rlm_evaluated"):
            result[column] = pd.array(result[column], dtype="boolean")
    return result


def _flag(frame: pd.DataFrame, name: str) -> pd.Series:
    for column in (f"i_{name}", f"flag_{name}"):
        if column in frame:
            return frame[column].fillna(0).astype(bool)
    if name == "ho_success" and "i_ho_exec_end" in frame:
        return frame["i_ho_exec_end"].fillna(0).astype(bool)
    return pd.Series(False, index=frame.index, dtype=bool)


def _identity(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in IDENTITY_COLUMNS:
        if column in frame:
            values = frame[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(
                    f"trajectory has non-constant identity column {column!r}"
                )
            result[column] = values.iloc[0]
        else:
            result[column] = None
    return result


def _cell_value(row: Mapping[str, Any], prefix: str, cell: int | None) -> float | None:
    if cell is None or cell < 0:
        return None
    value = row.get(f"{prefix}_{cell}")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _state_payload(
    row: Mapping[str, Any], *, source_cell: int | None = None
) -> dict[str, Any]:
    pcell_raw = row.get("pcell")
    pcell = None if pcell_raw is None or pd.isna(pcell_raw) else int(pcell_raw)
    target_raw = row.get("target_cell")
    target = None if target_raw is None or pd.isna(target_raw) else int(target_raw)
    source = pcell if source_cell is None else source_cell
    return {
        "pcell": pcell,
        "source_cell": source,
        "target_cell": target,
        "state": row.get("state"),
        "up_connected": row.get("up_connected", row.get("i_connected")),
        "rlm_evaluated": row.get("rlm_evaluated"),
        **{
            f"source_{name}": _cell_value(row, prefix, source)
            for name, prefix in _SIGNALS.items()
        },
        **{
            f"target_{name}": _cell_value(row, prefix, target)
            for name, prefix in _SIGNALS.items()
        },
    }


def derive_rrc_events(steps: pd.DataFrame, *, t_res_ms: int) -> pd.DataFrame:
    """Return one long-form row per asserted RRC event flag."""
    if t_res_ms <= 0:
        raise ValueError("t_res_ms must be positive")
    frame = steps.sort_values("time_step").reset_index(drop=True)
    identity = _identity(frame)
    output: list[dict[str, Any]] = []
    for flag, event_type in EVENT_FLAGS.items():
        for index in np.flatnonzero(_flag(frame, flag).to_numpy()):
            row = frame.iloc[int(index)]
            step = int(row["time_step"])
            output.append(
                {
                    **identity,
                    "event_type": event_type,
                    "time_step": step,
                    "time_ms": step * t_res_ms,
                    **_state_payload(row),
                }
            )
    columns = [
        *IDENTITY_COLUMNS,
        "event_type",
        "time_step",
        "time_ms",
        "pcell",
        "source_cell",
        "target_cell",
        "state",
        "up_connected",
        "rlm_evaluated",
        *[f"source_{name}" for name in _SIGNALS],
        *[f"target_{name}" for name in _SIGNALS],
    ]
    if not output:
        return _stable_derived_dtypes(pd.DataFrame(columns=columns))
    result = (
        pd.DataFrame(output)
        .sort_values(["time_step", "event_type"])
        .reset_index(drop=True)
    )
    return _stable_derived_dtypes(result)


def _phase_signals(
    row: Mapping[str, Any], phase: str, cell: int | None
) -> dict[str, Any]:
    return {
        f"{phase}_rlm_evaluated": row.get("rlm_evaluated"),
        **{
            f"{phase}_{name}": _cell_value(row, prefix, cell)
            for name, prefix in _SIGNALS.items()
        },
    }


def derive_handover_attempts(steps: pd.DataFrame, *, t_res_ms: int) -> pd.DataFrame:
    """Pair state-machine events into one row per preparation attempt."""
    if t_res_ms <= 0:
        raise ValueError("t_res_ms must be positive")
    frame = steps.sort_values("time_step").reset_index(drop=True)
    identity = _identity(frame)
    trajectory_id = identity.get("trajectory_id") or (
        f"{identity.get('method')}:{identity.get('split')}:"
        f"{identity.get('lambda')}:{identity.get('episode_idx')}:{identity.get('ue_id')}"
    )
    last_ttt_start: int | None = None
    last_ttt_expire: int | None = None
    current: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []

    def close(outcome: str, row: Mapping[str, Any], step: int) -> None:
        nonlocal current
        if current is None:
            raise RuntimeError(f"{outcome} at step {step} has no open handover attempt")
        current["outcome"] = outcome
        current["failure_cause"] = outcome if outcome.startswith("hof_") else None
        current["censored_phase"] = None
        current["_end_step"] = step
        if outcome == "success" and current["exec_end_step"] is None:
            current["exec_end_step"] = step
        output.append(current)
        current = None

    def advance_open_attempt(
        index: int,
        row: Mapping[str, Any],
        step: int,
        *,
        phase_start: bool = True,
        phase_end: bool = True,
    ) -> None:
        nonlocal current
        if current is None:
            return
        if phase_start and _flag(frame, "prep_expire").iloc[index]:
            current["prep_end_step"] = step
            current.update(_phase_signals(row, "prep_end", int(current["source_cell"])))
        if phase_start and _flag(frame, "exec_start").iloc[index]:
            current["exec_start_step"] = step
            current.update(
                _phase_signals(row, "exec_start", int(current["target_cell"]))
            )
        if not phase_end:
            return
        if _flag(frame, "exec_expire").iloc[index]:
            current["exec_end_step"] = step
            current.update(_phase_signals(row, "exec_end", int(current["target_cell"])))
        for outcome in ("hof_prep", "hof_cmd", "hof_exec"):
            if _flag(frame, outcome).iloc[index]:
                close(outcome, row, step)
                break
        if current is not None and _flag(frame, "ho_success").iloc[index]:
            close("success", row, step)

    for index, row in frame.iterrows():
        step = int(row["time_step"])
        if _flag(frame, "ttt_start").iloc[index]:
            last_ttt_start = step
        if _flag(frame, "ttt_expire").iloc[index]:
            last_ttt_expire = step

        # Mobility timers expire before autonomous A3 event processing in
        # RRCStepUE.step().  Consequently, one sample may both finish the
        # current execution (exec_expire + ho_success) and, for TTT=0, start
        # the next preparation.  Finish the attempt that was already open at
        # the start of the sample before assigning prep_start to a new one.
        had_open_attempt = current is not None
        starts_attempt = bool(_flag(frame, "prep_start").iloc[index])
        if had_open_attempt:
            # With zero preparation, the next attempt can also start execution
            # here. Its prep expiry / exec start must not overwrite the old one.
            advance_open_attempt(index, row, step, phase_start=not starts_attempt)

        if starts_attempt:
            if current is not None:
                raise RuntimeError(
                    f"prep_start at step {step} overlaps an open attempt"
                )
            source = int(row["pcell"])
            target_raw = row.get("target_cell")
            target = (
                None if target_raw is None or pd.isna(target_raw) else int(target_raw)
            )
            current = {
                **identity,
                "attempt_id": f"{trajectory_id}:attempt{len(output):04d}",
                "ttt_start_step": last_ttt_start,
                "ttt_expire_step": last_ttt_expire,
                "prep_start_step": step,
                "prep_end_step": None,
                "exec_start_step": None,
                "exec_end_step": None,
                "source_cell": source,
                "target_cell": target,
                **_phase_signals(row, "prep_start", source),
            }
        if current is None:
            continue
        if starts_attempt:
            # Execution is strictly positive. Any completion already handled
            # above belongs to the previous attempt, not this new preparation.
            advance_open_attempt(index, row, step, phase_end=not had_open_attempt)

    if current is not None:
        current["outcome"] = "censored"
        current["failure_cause"] = None
        current["censored_phase"] = (
            "exec" if current["exec_start_step"] is not None else "prep"
        )
        current["_end_step"] = int(frame["time_step"].iloc[-1])
        output.append(current)

    signal_columns = [
        f"{phase}_{name}"
        for phase in ("prep_start", "prep_end", "exec_start", "exec_end")
        for name in ("rlm_evaluated", *_SIGNALS)
    ]
    columns = [*ATTEMPT_COLUMNS, *signal_columns]
    if not output:
        return _stable_derived_dtypes(pd.DataFrame(columns=columns))
    result = pd.DataFrame(output)
    for prefix, start, end in (
        ("prep", "prep_start_step", "prep_end_step"),
        ("exec", "exec_start_step", "exec_end_step"),
        ("total", "prep_start_step", "_end_step"),
    ):
        samples = result[end] - result[start]
        result[f"{prefix}_duration_samples"] = samples.astype("Int64")
        result[f"{prefix}_duration_s"] = samples * t_res_ms / 1000.0
    return _stable_derived_dtypes(result.reindex(columns=columns))


def derive_outage_decomposition(
    steps: pd.DataFrame, attempts: pd.DataFrame, *, t_res_ms: int
) -> pd.DataFrame:
    """Aggregate each disconnected sample into exactly one causal segment."""
    if t_res_ms <= 0:
        raise ValueError("t_res_ms must be positive")
    frame = steps.sort_values("time_step").reset_index(drop=True)
    identity = _identity(frame)
    disconnected = ~frame["i_connected"].astype(bool).to_numpy()
    exec_running = frame["i_ho_exec_running"].astype(bool).to_numpy()
    rlf_running = frame["i_rlf_running"].astype(bool).to_numpy()
    segments = np.full(len(frame), "", dtype=object)

    for _, attempt in attempts.iterrows():
        start = attempt.get("exec_start_step")
        if pd.isna(start):
            continue
        end = attempt.get("exec_end_step")
        stop = len(frame) - 1 if pd.isna(end) else int(end)
        mask = (
            (frame["time_step"].to_numpy() >= int(start))
            & (frame["time_step"].to_numpy() <= stop)
            & exec_running
        )
        segments[mask] = (
            "successful_ho_exec" if attempt["outcome"] == "success" else "hof_related"
        )

    active_cause: str | None = None
    hof_subcause_counts = {name: 0 for name in ("hof_prep", "hof_cmd", "hof_exec")}
    for index in range(len(frame)):
        causes = [
            name
            for name in ("rlf_no_ho", "hof_prep", "hof_cmd", "hof_exec")
            if _flag(frame, name).iloc[index]
        ]
        if causes:
            if len(causes) != 1:
                raise RuntimeError(f"multiple RLF causes at step {index}: {causes}")
            active_cause = causes[0]
            if active_cause.startswith("hof_"):
                hof_subcause_counts[active_cause] += 1
        if rlf_running[index]:
            if active_cause is None:
                segments[index] = "unclassified_or_censored"
            else:
                segments[index] = (
                    "hof_related" if active_cause.startswith("hof_") else "rlf_no_ho"
                )
        elif active_cause is not None:
            active_cause = None

    segments[disconnected & (segments == "")] = "unclassified_or_censored"
    segments[~disconnected] = ""
    counts = {
        name: int(np.sum(segments == name))
        for name in (
            "successful_ho_exec",
            "hof_related",
            "rlf_no_ho",
            "unclassified_or_censored",
        )
    }
    total = int(disconnected.sum())
    if sum(counts.values()) != total:
        raise RuntimeError(
            "outage decomposition does not conserve disconnected samples"
        )
    return pd.DataFrame(
        [
            {
                **identity,
                **{f"{name}_samples": value for name, value in counts.items()},
                **{
                    f"{name}_seconds": value * t_res_ms / 1000.0
                    for name, value in counts.items()
                },
                "outage_samples": total,
                "outage_seconds": total * t_res_ms / 1000.0,
                "simulated_seconds": len(frame) * t_res_ms / 1000.0,
                "outage_fraction": total / len(frame),
                **{f"num_{name}": value for name, value in hof_subcause_counts.items()},
            }
        ]
    )


__all__ = [
    "ATTEMPT_COLUMNS",
    "EVENT_FLAGS",
    "IDENTITY_COLUMNS",
    "derive_handover_attempts",
    "derive_outage_decomposition",
    "derive_rrc_events",
]
