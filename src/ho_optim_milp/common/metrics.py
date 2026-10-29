"""Metrics shared by offline MILP and reference trajectories."""

from __future__ import annotations

from typing import Any, Mapping

from mobile_network_sim.rrc.flags import HANDOVER_FAILURE_CAUSES, LINK_FAILURE_CAUSES
from mobile_network_sim.rrc.rrc_utils import count_ping_pongs

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


CAUSE_TO_COLUMN = {cause: f"num_{cause}" for cause in LINK_FAILURE_CAUSES}


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
