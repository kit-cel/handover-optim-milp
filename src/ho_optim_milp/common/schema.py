"""Canonical column names for generated MILP/A3 result artifacts.

Single source of truth, owned by the public dataset producer. Consumers may
import these names instead of maintaining a second copy; a previous split
between ``pcell`` and ``pcell_gurobi`` required a dataset-wide repair.

One name per concept
--------------------
Every method writes the *same* column name for the same quantity, even
where only one method can produce a non-trivial value. A MILP row and an
A3 row must be concatenable and joinable without translation:

* the UE index is ``ue_id`` everywhere (never ``ue_idx``),
* A3 parameters are ``ttt_ms``/``hys_db``/``offset_db`` everywhere
  (never ``a3_*``), matching what the released ``lambda_winners.csv``
  already uses and what ``handover_rl`` already reads,
* an undefined ratio is ``NaN`` everywhere (never ``inf``),
* handover-failure counts exist on every method; the MILP simply reports
  zero, because its formulation cannot produce a failed handover.
"""

from __future__ import annotations

from typing import Iterable

# --------------------------------------------------------------------------
# Identity: present on every result row, whatever produced it.
# --------------------------------------------------------------------------

METHOD_COLUMN = "method"
SPLIT_COLUMN = "split"
MILP_EPISODE_COLUMN = "episode_idx"
MILP_UE_COLUMN = "ue_id"
MILP_LAMBDA_COLUMN = "lambda"
MILP_TIME_STEP_COLUMN = "time_step"

#: Identifying columns every result row must carry.
IDENTITY_COLUMNS = (
    METHOD_COLUMN,
    SPLIT_COLUMN,
    MILP_EPISODE_COLUMN,
    MILP_UE_COLUMN,
    MILP_LAMBDA_COLUMN,
)

# Harmonized to "pcell" across all splits. The ID splits historically used
# "pcell_gurobi" for the same concept and were renamed in place; new
# releases must never reintroduce that divergence.
MILP_PCELL_COLUMN = "pcell"

# --------------------------------------------------------------------------
# A3 parameters. Unprefixed, matching the released lambda_winners.csv.
# --------------------------------------------------------------------------

A3_TTT_COLUMN = "ttt_ms"
A3_HYS_COLUMN = "hys_db"
A3_OFFSET_COLUMN = "offset_db"
A3_LAMBDA_COLUMN = MILP_LAMBDA_COLUMN

#: A3 parameter columns, in reporting order.
A3_PARAMETER_COLUMNS = (A3_TTT_COLUMN, A3_HYS_COLUMN, A3_OFFSET_COLUMN)

# --------------------------------------------------------------------------
# Aggregated metrics, written identically by every method.
# --------------------------------------------------------------------------

METRIC_SIMULATED_TIME_COLUMN = "simulated_time_s"
METRIC_SIMULATED_STEPS_COLUMN = "simulated_steps"
METRIC_MEAN_CAPACITY_COLUMN = "mean_capacity"
METRIC_MAX_MEAN_CAPACITY_COLUMN = "max_mean_capacity"
METRIC_REL_CONNECTED_TIME_COLUMN = "rel_connected_time"
METRIC_NUM_HO_COLUMN = "num_ho"
METRIC_HO_PER_S_COLUMN = "ho_per_s"
METRIC_NUM_PP_COLUMN = "num_pp"
METRIC_PP_PER_S_COLUMN = "pp_per_s"
METRIC_PP_RATE_COLUMN = "pp_rate"
METRIC_NUM_RLF_COLUMN = "num_rlf"
METRIC_RLF_PER_S_COLUMN = "rlf_per_s"
METRIC_NUM_HOF_COLUMN = "num_hof"
METRIC_HOF_RATE_COLUMN = "hof_rate"
METRIC_OBJECTIVE_COLUMN = "ho_objective_value"

# Link-failure breakdown, mirroring
# :data:`mobile_network_sim.rrc.flags.LINK_FAILURE_CAUSES`. Splitting the
# RLF count by cause is what makes an A3-versus-policy comparison
# diagnosable: handing over too eagerly and too late both raise num_rlf,
# but they raise *different* causes.
METRIC_NUM_RLF_NO_HO_COLUMN = "num_rlf_no_ho"
METRIC_NUM_HOF_PREP_COLUMN = "num_hof_prep"
METRIC_NUM_HOF_CMD_COLUMN = "num_hof_cmd"
METRIC_NUM_HOF_EXEC_COLUMN = "num_hof_exec"

#: Per-cause link-failure counts, in reporting order. These sum to
#: :data:`METRIC_NUM_RLF_COLUMN`.
LINK_FAILURE_COUNT_COLUMNS = (
    METRIC_NUM_RLF_NO_HO_COLUMN,
    METRIC_NUM_HOF_PREP_COLUMN,
    METRIC_NUM_HOF_CMD_COLUMN,
    METRIC_NUM_HOF_EXEC_COLUMN,
)

#: The handover-failure subset; these sum to :data:`METRIC_NUM_HOF_COLUMN`.
HANDOVER_FAILURE_COUNT_COLUMNS = (
    METRIC_NUM_HOF_PREP_COLUMN,
    METRIC_NUM_HOF_CMD_COLUMN,
    METRIC_NUM_HOF_EXEC_COLUMN,
)

#: Every aggregated metric column, in reporting order. Each method emits
#: all of them, so a merged table has no method-dependent holes.
METRIC_COLUMNS = (
    METRIC_SIMULATED_TIME_COLUMN,
    METRIC_SIMULATED_STEPS_COLUMN,
    METRIC_MEAN_CAPACITY_COLUMN,
    METRIC_MAX_MEAN_CAPACITY_COLUMN,
    METRIC_REL_CONNECTED_TIME_COLUMN,
    METRIC_NUM_HO_COLUMN,
    METRIC_HO_PER_S_COLUMN,
    METRIC_NUM_PP_COLUMN,
    METRIC_PP_PER_S_COLUMN,
    METRIC_PP_RATE_COLUMN,
    METRIC_NUM_RLF_COLUMN,
    METRIC_RLF_PER_S_COLUMN,
    METRIC_NUM_HOF_COLUMN,
    METRIC_HOF_RATE_COLUMN,
    *LINK_FAILURE_COUNT_COLUMNS,
    METRIC_OBJECTIVE_COLUMN,
)

# --------------------------------------------------------------------------
# Per-step trajectory columns.
# --------------------------------------------------------------------------

STEP_SERVING_SINR_COLUMN = "serving_sinr_db"
STEP_SERVING_CAPACITY_COLUMN = "serving_capacity"
STEP_DELIVERED_CAPACITY_COLUMN = "delivered_capacity"
STEP_CONNECTED_COLUMN = "i_connected"
STEP_HO_EXEC_RUNNING_COLUMN = "i_ho_exec_running"
STEP_HO_EXEC_END_COLUMN = "i_ho_exec_end"
STEP_RLF_RUNNING_COLUMN = "i_rlf_running"
STEP_RLF_START_COLUMN = "i_rlf_start"

#: Per-step columns every method writes, with identical meaning. This is
#: the slice a consumer can compare across methods without knowing which
#: produced it -- and the slice BC/AIRL read as expert demonstrations.
STEP_SHARED_COLUMNS = (
    MILP_TIME_STEP_COLUMN,
    MILP_PCELL_COLUMN,
    STEP_SERVING_SINR_COLUMN,
    STEP_SERVING_CAPACITY_COLUMN,
    STEP_DELIVERED_CAPACITY_COLUMN,
    STEP_CONNECTED_COLUMN,
    STEP_HO_EXEC_RUNNING_COLUMN,
    STEP_HO_EXEC_END_COLUMN,
    STEP_RLF_RUNNING_COLUMN,
    STEP_RLF_START_COLUMN,
)


# --------------------------------------------------------------------------
# Per-cell trajectory columns.
# --------------------------------------------------------------------------
#
# The full per-cell radio view is stored alongside the serving-cell
# scalars so a solved trajectory can be re-checked, and so a consumer can
# rank neighbour cells, without re-reading and re-filtering the episode
# HDF5. Each quantity is flattened into one column per cell rather than a
# nested list column, so readers can access individual cells directly.
#
# The naming is defined here rather than inline at the write site so the
# readers (the offline analyses rank neighbours by exactly these columns)
# cannot drift from the writer -- the same reasoning that makes every
# other name in this module producer-owned.

STEP_PER_CELL_SINR_PREFIX = "sinr_db"
STEP_PER_CELL_CAPACITY_PREFIX = "capacity"
STEP_PER_CELL_L3_RSRP_PREFIX = "l3_rsrp_dbm"
STEP_PER_CELL_L3_SINR_PREFIX = "l3_sinr_db"
STEP_PER_CELL_L1_QIN_SINR_PREFIX = "l1_qin_sinr_db"
STEP_PER_CELL_L1_QOUT_SINR_PREFIX = "l1_qout_sinr_db"
STEP_PER_CELL_QIN_PREFIX = "q_in"
STEP_PER_CELL_QOUT_PREFIX = "q_out"


def per_cell_sinr_column(cell: int) -> str:
    """Column holding cell ``cell``'s SINR in dB.

    Whether values are raw or filtered is a property of the dataset, not
    of the column name. Consumers must read the dataset metadata before
    interpreting these values.
    """
    return f"{STEP_PER_CELL_SINR_PREFIX}_{cell}"


def per_cell_capacity_column(cell: int) -> str:
    """Column holding cell ``cell``'s Shannon capacity in bps/Hz."""
    return f"{STEP_PER_CELL_CAPACITY_PREFIX}_{cell}"


def per_cell_l3_rsrp_column(cell: int) -> str:
    return f"{STEP_PER_CELL_L3_RSRP_PREFIX}_{cell}"


def per_cell_l3_sinr_column(cell: int) -> str:
    return f"{STEP_PER_CELL_L3_SINR_PREFIX}_{cell}"


def per_cell_l1_qin_sinr_column(cell: int) -> str:
    return f"{STEP_PER_CELL_L1_QIN_SINR_PREFIX}_{cell}"


def per_cell_l1_qout_sinr_column(cell: int) -> str:
    return f"{STEP_PER_CELL_L1_QOUT_SINR_PREFIX}_{cell}"


def per_cell_qin_column(cell: int) -> str:
    return f"{STEP_PER_CELL_QIN_PREFIX}_{cell}"


def per_cell_qout_column(cell: int) -> str:
    return f"{STEP_PER_CELL_QOUT_PREFIX}_{cell}"


def count_per_cell_columns(columns: Iterable[str], prefix: str) -> int:
    """Return how many cells ``columns`` carries under ``prefix``.

    Derives the cell count from the data instead of hardcoding it, and
    verifies the columns form a gap-free ``0..n-1`` range -- a missing
    middle index would otherwise silently shift every neighbour ranking
    built from them.

    Raises
    ------
    ValueError
        If the indices present are not exactly ``0..n-1``.
    """
    found: set[int] = set()
    for name in columns:
        head, sep, tail = name.rpartition("_")
        if sep and head == prefix and tail.isdigit():
            found.add(int(tail))
    if not found:
        return 0
    expected = set(range(max(found) + 1))
    if found != expected:
        missing = sorted(expected - found)
        raise ValueError(
            f"per-cell columns for prefix {prefix!r} are not a gap-free 0..n-1 "
            f"range: missing indices {missing}"
        )
    return len(found)
