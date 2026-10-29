"""Metrics shared by offline MILP and reference trajectories."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .schema import (
    METRIC_HOF_RATE_COLUMN,
    METRIC_HO_PER_S_COLUMN,
    METRIC_MAX_MEAN_CAPACITY_COLUMN,
    METRIC_MEAN_CAPACITY_COLUMN,
    METRIC_NUM_HOF_COLUMN,
    METRIC_NUM_HO_COLUMN,
    METRIC_NUM_PP_COLUMN,
    METRIC_NUM_RLF_COLUMN,
    METRIC_OBJECTIVE_COLUMN,
    METRIC_PP_PER_S_COLUMN,
    METRIC_PP_RATE_COLUMN,
    METRIC_REL_CONNECTED_TIME_COLUMN,
    METRIC_RLF_PER_S_COLUMN,
    METRIC_SIMULATED_STEPS_COLUMN,
    METRIC_SIMULATED_TIME_COLUMN,
    MILP_UE_COLUMN,
)


LINK_FAILURE_CAUSES = ("rlf_no_ho", "hof_prep", "hof_cmd", "hof_exec")
HANDOVER_FAILURE_CAUSES = ("hof_prep", "hof_cmd", "hof_exec")
CAUSE_TO_COLUMN = {cause: f"num_{cause}" for cause in LINK_FAILURE_CAUSES}


def count_ping_pongs(
    pcell: NDArray[np.integer],
    ho_completed: NDArray[np.integer] | NDArray[np.bool_],
    t_mts_steps: int,
) -> int:
    """Count A-B-A handovers completed within the inclusive time window."""
    cells = np.asarray(pcell)
    events = np.asarray(ho_completed)
    if cells.ndim != 1 or events.ndim != 1 or cells.shape != events.shape:
        raise ValueError("pcell and ho_completed must be matching 1-D arrays")
    if t_mts_steps < 0:
        raise ValueError("t_mts_steps must be non-negative")
    completions = np.flatnonzero(events)
    return sum(
        int(first > 0 and second - first <= t_mts_steps and cells[first - 1] == cells[second])
        for first, second in zip(completions[:-1], completions[1:])
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Return NaN for an undefined ratio."""
    return numerator / denominator if denominator > 0 else float("nan")


def build_metric_row(
    *,
    ue_id: int,
    simulated_time_s: float,
    simulated_steps: int,
    mean_capacity: float,
    max_mean_capacity: float,
    rel_connected_time: float,
    num_ho: int,
    num_pp: int,
    num_rlf: int,
    link_failure_counts: Mapping[str, int] | None,
    objective_value: float,
) -> dict[str, Any]:
    """Build canonical MILP or reference metrics without a simulator import."""
    if link_failure_counts is None:
        counts = {cause: 0 for cause in LINK_FAILURE_CAUSES}
        counts["rlf_no_ho"] = int(num_rlf)
    else:
        counts = {cause: int(link_failure_counts.get(cause, 0)) for cause in LINK_FAILURE_CAUSES}
    if sum(counts.values()) != int(num_rlf):
        raise ValueError("link failure causes do not sum to num_rlf")
    num_hof = sum(counts[cause] for cause in HANDOVER_FAILURE_CAUSES)
    row: dict[str, Any] = {
        MILP_UE_COLUMN: int(ue_id),
        METRIC_SIMULATED_TIME_COLUMN: float(simulated_time_s),
        METRIC_SIMULATED_STEPS_COLUMN: int(simulated_steps),
        METRIC_MEAN_CAPACITY_COLUMN: float(mean_capacity),
        METRIC_MAX_MEAN_CAPACITY_COLUMN: float(max_mean_capacity),
        METRIC_REL_CONNECTED_TIME_COLUMN: float(rel_connected_time),
        METRIC_NUM_HO_COLUMN: int(num_ho),
        METRIC_HO_PER_S_COLUMN: _ratio(num_ho, simulated_time_s),
        METRIC_NUM_PP_COLUMN: int(num_pp),
        METRIC_PP_PER_S_COLUMN: _ratio(num_pp, simulated_time_s),
        METRIC_PP_RATE_COLUMN: _ratio(num_pp, num_ho),
        METRIC_NUM_RLF_COLUMN: int(num_rlf),
        METRIC_RLF_PER_S_COLUMN: _ratio(num_rlf, simulated_time_s),
        METRIC_NUM_HOF_COLUMN: int(num_hof),
        METRIC_HOF_RATE_COLUMN: _ratio(num_hof, num_hof + num_ho),
        METRIC_OBJECTIVE_COLUMN: float(objective_value),
    }
    row.update({CAUSE_TO_COLUMN[cause]: counts[cause] for cause in LINK_FAILURE_CAUSES})
    return row
