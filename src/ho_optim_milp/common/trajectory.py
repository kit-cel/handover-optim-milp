"""Build per-step MILP and A3 trajectory tables with shared column names."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .schema import (
    MILP_PCELL_COLUMN,
    MILP_TIME_STEP_COLUMN,
    STEP_CONNECTED_COLUMN,
    STEP_DELIVERED_CAPACITY_COLUMN,
    STEP_HO_EXEC_END_COLUMN,
    STEP_HO_EXEC_RUNNING_COLUMN,
    STEP_RLF_RUNNING_COLUMN,
    STEP_RLF_START_COLUMN,
    STEP_SERVING_CAPACITY_COLUMN,
    STEP_SERVING_SINR_COLUMN,
    per_cell_capacity_column,
    per_cell_l1_qin_sinr_column,
    per_cell_l1_qout_sinr_column,
    per_cell_l3_rsrp_column,
    per_cell_l3_sinr_column,
    per_cell_qin_column,
    per_cell_qout_column,
    per_cell_sinr_column,
)

# Store MILP auxiliaries so saved solutions can be checked against constraints.
_MILP_SCALAR_EXTRAS = (
    "i_q_in",
    "i_q_out",
    "u_n310_raw",
    "u_n310_sat",
    "c_n310",
    "i_n310",
    "n310_b_reset",
    "n310_b_q",
    "n310_b_c",
    "u_n311_raw",
    "u_n311_sat",
    "c_n311",
    "i_n311",
    "n311_b_reset",
    "n311_b_q",
    "n311_b_c",
    "i_t310_start",
    "i_t310_stop",
    "i_t310_cancel_due_to_n311",
    "i_t310_expiry",
    "tau_t310_rem",
    "i_t310_eq_1",
    "i_t310_active",
    "i_t310_b_mode_rlf",
    "i_t310_b_mode_start",
    "i_t310_b_mode_stop",
    "i_t310_b_mode_decr",
    "i_rlf_in_last_t_ho_steps",
    "i_ho_cell_change",
    "i_rate_conn_aux",
    "s_rate",
    "r_rate",
)

# Per-cell arrays, expanded to one column per cell so the frame stays
# flat and parquet-friendly rather than holding list-valued cells.
_MILP_PER_CELL_EXTRAS = ("x_bs", "i_ho_same_cell")


def _shared_frame_columns(
    *,
    pcell: NDArray[np.integer],
    serving_sinr_db: NDArray[np.floating],
    ho_exec_running: NDArray[np.integer],
    ho_exec_end: NDArray[np.integer],
    rlf_running: NDArray[np.integer],
    rlf_start: NDArray[np.integer],
) -> dict[str, NDArray[Any]]:
    """Build the columns every method shares, from its own raw arrays.

    Capacity and connectivity are derived here rather than taken from
    each method's own bookkeeping, so the two cannot disagree on what
    "delivered capacity" means.
    """
    n_steps = int(pcell.shape[0])
    serving_capacity = np.log2(1.0 + 10.0 ** (serving_sinr_db / 10.0))
    outage = (ho_exec_running + rlf_running) > 0
    return {
        MILP_TIME_STEP_COLUMN: np.arange(n_steps, dtype=np.int32),
        MILP_PCELL_COLUMN: pcell.astype(np.int32, copy=False),
        STEP_SERVING_SINR_COLUMN: serving_sinr_db.astype(np.float32, copy=False),
        STEP_SERVING_CAPACITY_COLUMN: serving_capacity.astype(np.float32, copy=False),
        STEP_DELIVERED_CAPACITY_COLUMN: np.where(outage, 0.0, serving_capacity).astype(
            np.float32
        ),
        STEP_CONNECTED_COLUMN: (~outage).astype(np.int8),
        STEP_HO_EXEC_RUNNING_COLUMN: ho_exec_running.astype(np.int8, copy=False),
        STEP_HO_EXEC_END_COLUMN: ho_exec_end.astype(np.int8, copy=False),
        STEP_RLF_RUNNING_COLUMN: rlf_running.astype(np.int8, copy=False),
        STEP_RLF_START_COLUMN: rlf_start.astype(np.int8, copy=False),
    }


def milp_step_frame(results: Mapping[str, np.ndarray]) -> Any:
    """Convert one MILP solution into a per-step trajectory table."""
    pcell = np.asarray(results[MILP_PCELL_COLUMN])
    sinr = np.asarray(results["sinr_db"])
    serving_sinr = sinr[pcell, np.arange(pcell.shape[0])]

    columns = _shared_frame_columns(
        pcell=pcell,
        serving_sinr_db=serving_sinr,
        ho_exec_running=np.asarray(results["i_ho_exec_running"]),
        ho_exec_end=np.asarray(results["i_ho_exec_end"]),
        rlf_running=np.asarray(results["i_rlf_running"]),
        rlf_start=np.asarray(results["i_rlf_start"]),
    )
    for name in _MILP_SCALAR_EXTRAS:
        if name in results:
            columns[name] = np.asarray(results[name])
    for name in _MILP_PER_CELL_EXTRAS:
        if name not in results:
            continue
        matrix = np.asarray(results[name])
        for cell in range(matrix.shape[0]):
            columns[f"{name}_{cell}"] = matrix[cell]
    # Per-cell SINR and capacity: the model inputs, kept so a stored
    # solution can be re-checked without re-reading and re-filtering the
    # episode HDF5.
    capacity = np.asarray(results["capacity"])
    for cell in range(sinr.shape[0]):
        columns[per_cell_sinr_column(cell)] = sinr[cell].astype(np.float32, copy=False)
        columns[per_cell_capacity_column(cell)] = capacity[cell].astype(
            np.float32, copy=False
        )
    return pd.DataFrame(columns)


def reference_step_frame(
    flags: Mapping[str, np.ndarray],
    sinr_db: np.ndarray,
    ue_index: int,
    *,
    snapshots: Sequence[Mapping[str, object]] | None = None,
    l3_rsrp_dbm: np.ndarray | None = None,
    l3_sinr_db: np.ndarray | None = None,
    l1_qin_sinr_db: np.ndarray | None = None,
    l1_qout_sinr_db: np.ndarray | None = None,
    q_in: np.ndarray | None = None,
    q_out: np.ndarray | None = None,
) -> Any:
    """Convert one A3 run into a per-step trajectory table."""
    pcell = np.asarray(flags["pcell"])[:, ue_index]
    serving_sinr = sinr_db[pcell, ue_index, np.arange(pcell.shape[0])]

    columns = _shared_frame_columns(
        pcell=pcell,
        serving_sinr_db=serving_sinr,
        ho_exec_running=np.asarray(flags["ho_exec_running"])[:, ue_index],
        ho_exec_end=np.asarray(flags["ho_exec_end"])[:, ue_index],
        rlf_running=np.asarray(flags["rlf_running"])[:, ue_index],
        rlf_start=np.asarray(flags["rlf_start"])[:, ue_index],
    )
    # Every remaining recorded flag, under its own name. These are the
    # RRC internals (TTT/prep/exec/T310 events and the failure causes)
    # that make an A3-versus-policy comparison diagnosable.
    for name, array in flags.items():
        if name in ("pcell", "ho_exec_running", "ho_exec_end", "rlf_running", "rlf_start"):
            continue
        columns[f"i_{name}"] = np.asarray(array)[:, ue_index].astype(np.int8, copy=False)
    if snapshots is not None:
        if len(snapshots) != pcell.shape[0]:
            raise ValueError("snapshot count must match trajectory length")
        snapshot_names = [
            name
            for name in snapshots[0]
            if not name.startswith("flag_")
            and name not in ("time_ms", "imsi", "pcell")
            and not isinstance(snapshots[0][name], (list, tuple, dict, np.ndarray))
        ]
        for name in snapshot_names:
            columns[name] = np.asarray([snapshot[name] for snapshot in snapshots])
    for cell in range(sinr_db.shape[0]):
        columns[per_cell_sinr_column(cell)] = sinr_db[cell, ue_index].astype(
            np.float32, copy=False
        )
    optional_per_cell = (
        (l3_rsrp_dbm, per_cell_l3_rsrp_column, np.float32),
        (l3_sinr_db, per_cell_l3_sinr_column, np.float32),
        (l1_qin_sinr_db, per_cell_l1_qin_sinr_column, np.float32),
        (l1_qout_sinr_db, per_cell_l1_qout_sinr_column, np.float32),
        (q_in, per_cell_qin_column, np.int8),
        (q_out, per_cell_qout_column, np.int8),
    )
    for array, column_name, dtype in optional_per_cell:
        if array is None:
            continue
        values = np.asarray(array)
        if values.shape != (pcell.shape[0], sinr_db.shape[0]):
            raise ValueError(
                "per-cell reference array must have shape "
                f"({pcell.shape[0]}, {sinr_db.shape[0]}), got {values.shape}"
            )
        for cell in range(values.shape[1]):
            columns[column_name(cell)] = values[:, cell].astype(dtype, copy=False)
    return pd.DataFrame(columns)


__all__ = ["milp_step_frame", "reference_step_frame"]
