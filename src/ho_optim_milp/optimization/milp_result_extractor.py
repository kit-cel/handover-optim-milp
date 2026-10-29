"""Result extraction and reporting for a solved MILP handover optimizer."""

import csv
import os
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING

import numpy as np

from ..common.metrics import build_metric_row, count_ping_pongs
from ..common.schema import MILP_PCELL_COLUMN

if TYPE_CHECKING:
    from .milp_variables import ModelVars
    from .milp_dataclasses import ProblemData, RRCConstants


def _binary_solution(values: Any, threshold: float = 0.5) -> np.ndarray:
    """Convert Gurobi binary values to exact integer flags."""
    values = np.asarray(values, dtype=float)
    return (values >= threshold).astype(np.int8)


def _integer_solution(values: Any) -> np.ndarray:
    """Convert Gurobi integer values to exact integer arrays."""
    values = np.asarray(values, dtype=float)
    return np.rint(values).astype(np.int64)


class ResultExtractor:
    """Extracts, computes, and logs results from a solved MILP model.

    Instantiate *after* optimization is complete.  All Gurobi ``.X`` reads
    happen once in ``__init__`` and are cached as plain NumPy arrays so that
    every subsequent metric or logging call is fast and free of Gurobi I/O.

    Parameters
    ----------
    vars : ModelVars
        Gurobi decision-variable containers for the solved model.
    data : ProblemData
        Problem input data (RSRP, SINR, dimensions, …).
    rrc_consts : RRCConstants
        RRC timer/counter constants, already converted to time-steps.
    simulation_id : str
        Unique simulation identifier (propagated into metrics dict).
    lambda_r : float
        Lagrange multiplier used in the objective.
    ep_name : str
        Episode name used for file naming.
    log_dir : str
        Root directory for CSV output.
    """

    def __init__(
        self,
        model_vars: "ModelVars",
        data: "ProblemData",
        rrc_consts: "RRCConstants",
        *,
        simulation_id: str,
        lambda_r: float,
        ep_name: str,
        log_dir: str,
    ) -> None:
        """Cache all Gurobi ``.X`` arrays from the solved model.

        Reading ``.X`` triggers a round-trip to the Gurobi solver object.
        Caching every array once here avoids repeated solver calls in the
        metric and snapshot methods that follow.
        """
        self._data = data
        self._rrc_consts = rrc_consts
        self._simulation_id = simulation_id
        self._lambda_r = lambda_r
        self._ep_name = ep_name
        self._log_dir = log_dir

        v = model_vars
        self._x_bs: np.ndarray = _binary_solution(v.x_bs.X)
        self._pcells: np.ndarray = np.argmax(self._x_bs, axis=0)
        self._i_q_in: np.ndarray = _binary_solution(v.i_q_in.X)
        self._i_q_out: np.ndarray = _binary_solution(v.i_q_out.X)
        self._n310_u_raw: np.ndarray = _integer_solution(v.n310.u_raw.X)
        self._n310_u_sat: np.ndarray = _integer_solution(v.n310.u_sat.X)
        self._n310_cnt: np.ndarray = _integer_solution(v.n310.cnt.X)
        self._n310_ind: np.ndarray = _binary_solution(v.n310.ind.X)
        self._n310_b_reset: np.ndarray = _binary_solution(v.n310.b_reset.X)
        self._n310_b_q: np.ndarray = _binary_solution(v.n310.b_q.X)
        self._n310_b_c: np.ndarray = _binary_solution(v.n310.b_c.X)
        self._n311_u_raw: np.ndarray = _integer_solution(v.n311.u_raw.X)
        self._n311_u_sat: np.ndarray = _integer_solution(v.n311.u_sat.X)
        self._n311_cnt: np.ndarray = _integer_solution(v.n311.cnt.X)
        self._n311_ind: np.ndarray = _binary_solution(v.n311.ind.X)
        self._n311_b_reset: np.ndarray = _binary_solution(v.n311.b_reset.X)
        self._n311_b_q: np.ndarray = _binary_solution(v.n311.b_q.X)
        self._n311_b_c: np.ndarray = _binary_solution(v.n311.b_c.X)
        self._t310_start: np.ndarray = _binary_solution(v.t310.start.X)
        self._t310_stop: np.ndarray = _binary_solution(v.t310.stop.X)
        self._t310_cancel_due_to_n311: np.ndarray = _binary_solution(
            v.t310.cancel_due_to_n311.X
        )
        self._t310_expiry: np.ndarray = _binary_solution(v.t310.expiry.X)
        self._t310_tau: np.ndarray = _integer_solution(v.t310.tau.X)
        self._t310_tau_eq_1: np.ndarray = _binary_solution(v.t310.tau_eq_1.X)
        self._t310_active: np.ndarray = _binary_solution(v.t310.active.X)
        self._t310_b_mode_rlf: np.ndarray = _binary_solution(v.t310.b_mode_rlf.X)
        self._t310_b_mode_start: np.ndarray = _binary_solution(v.t310.b_mode_start.X)
        self._t310_b_mode_stop: np.ndarray = _binary_solution(v.t310.b_mode_stop.X)
        self._t310_b_mode_decr: np.ndarray = _binary_solution(v.t310.b_mode_decr.X)
        self._rlf_start: np.ndarray = _binary_solution(v.rlf.start.X)
        self._rlf_running: np.ndarray = _binary_solution(v.rlf.running.X)
        self._rlf_in_last_t_ho_steps: np.ndarray = np.asarray(
            _binary_solution(v.rlf.in_last_t_ho_steps.X)
        )
        self._ho_same_cell: np.ndarray = _binary_solution(v.ho.same_cell.X)
        self._ho_cell_change: np.ndarray = _binary_solution(v.ho.cell_change.X)
        self._ho_exec_end: np.ndarray = _binary_solution(v.ho.exec_end.X)
        self._ho_exec_running: np.ndarray = _binary_solution(v.ho.exec_running.X)
        self._rate_conn_aux: np.ndarray = _binary_solution(v.rate.conn_aux.X)
        self._rate_s_rate: np.ndarray = np.asarray(v.rate.s_rate.X, dtype=float)
        self._rate_r_rate: np.ndarray = np.asarray(v.rate.r_rate.X, dtype=float)

    def extract_results(self) -> dict[str, np.ndarray]:
        """Return a flat dict of all key solution arrays."""
        return {
            "sinr_db": self._data.sinr_db,
            "capacity": self._data.capacity,
            "x_bs": self._x_bs,
            "pcell": self._pcells,
            "i_q_in": self._i_q_in,
            "i_q_out": self._i_q_out,
            "u_n310_raw": self._n310_u_raw,
            "u_n310_sat": self._n310_u_sat,
            "c_n310": self._n310_cnt,
            "i_n310": self._n310_ind,
            "n310_b_reset": self._n310_b_reset,
            "n310_b_q": self._n310_b_q,
            "n310_b_c": self._n310_b_c,
            "u_n311_raw": self._n311_u_raw,
            "u_n311_sat": self._n311_u_sat,
            "c_n311": self._n311_cnt,
            "i_n311": self._n311_ind,
            "n311_b_reset": self._n311_b_reset,
            "n311_b_q": self._n311_b_q,
            "n311_b_c": self._n311_b_c,
            "i_t310_start": self._t310_start,
            "i_t310_stop": self._t310_stop,
            "i_t310_cancel_due_to_n311": self._t310_cancel_due_to_n311,
            "i_t310_expiry": self._t310_expiry,
            "tau_t310_rem": self._t310_tau,
            "i_t310_eq_1": self._t310_tau_eq_1,
            "i_t310_b_mode_rlf": self._t310_b_mode_rlf,
            "i_t310_b_mode_start": self._t310_b_mode_start,
            "i_t310_b_mode_stop": self._t310_b_mode_stop,
            "i_t310_b_mode_decr": self._t310_b_mode_decr,
            "i_rlf_start": self._rlf_start,
            "i_rlf_in_last_t_ho_steps": self._rlf_in_last_t_ho_steps,
            "i_ho_exec_end": self._ho_exec_end,
            "i_ho_exec_running": self._ho_exec_running,
            "i_ho_cell_change": self._ho_cell_change,
            "i_ho_same_cell": self._ho_same_cell,
            "i_t310_active": self._t310_active,
            "i_rlf_running": self._rlf_running,
            "i_rate_conn_aux": self._rate_conn_aux,
            "s_rate": self._rate_s_rate,
            "r_rate": self._rate_r_rate,
        }

    def get_result_metrics(self, obj_value: float) -> dict[int, dict[str, Any]]:
        """Compute and return scalar performance metrics.

        Parameters
        ----------
        obj_value : float
            Optimal objective value from the solver (``model.ObjVal`` or
            ``obj_expr.getValue()``).
        """
        dt_s = self._data.n_steps * self._rrc_consts.t_res_ms / 1_000
        c_mean = self._get_mean_capacity()
        t_connected = self._get_connected_time()
        num_ho = int(np.sum(self._ho_exec_end))
        num_pp, pp_per_s = self._get_num_pp()

        # Built through the shared row builder so the MILP and the
        # reference simulation cannot drift on how an undefined ratio is
        # spelled, or on which columns exist at all -- they already had.
        # link_failure_counts=None marks the MILP: its formulation forbids
        # a cell change while an RLF is recent, so no handover can fail.
        row = build_metric_row(
            ue_id=self._data.ue_idx,
            simulated_time_s=dt_s,
            simulated_steps=self._data.n_steps,
            mean_capacity=c_mean,
            max_mean_capacity=float(np.mean(np.max(self._data.capacity, axis=0))),
            rel_connected_time=t_connected / self._data.n_steps,
            num_ho=num_ho,
            num_pp=num_pp,
            num_rlf=int(np.sum(self._rlf_start)),
            link_failure_counts=None,
            objective_value=obj_value,
        )
        row["simulation_id"] = self._simulation_id
        return {int(self._data.ue_idx): row}

    def log_results_to_csv(
        self,
        file_name: str | None = None,
        subfolder: str | None = None,
        extra_columns: dict[str, Any] | None = None,
    ) -> None:
        """Atomically replace a per-step snapshot CSV.

        Parameters
        ----------
        file_name : str, optional
            Output file name.  Auto-generated from simulation/episode/UE
            metadata when omitted.
        subfolder : str, optional
            Subdirectory inside ``log_dir`` for the file.
        extra_columns : dict[str, Any], optional
            Additional scalar metadata columns repeated on every row.
        """
        if file_name is None:
            file_name = (
                f"optim_{self._simulation_id}_"
                f"ep{self._ep_name}_ue{self._data.ue_idx}_lambda{self._lambda_r}.csv"
            )

        if subfolder is not None:
            full_log_path = os.path.join(
                self._log_dir, subfolder, "full_results", file_name
            )
        else:
            full_log_path = os.path.join(self._log_dir, "full_results", file_name)

        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)
        extra_columns = extra_columns or {}
        destination = Path(full_log_path)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                temp_path = Path(f.name)
                header = [*extra_columns.keys(), *self._get_snapshot(0).keys()]
                writer = csv.DictWriter(f, fieldnames=header, delimiter=";")
                writer.writeheader()
                for i in range(self._data.n_steps):
                    writer.writerow({**extra_columns, **self._get_snapshot(i)})
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        if not os.path.isfile(full_log_path):
            raise OSError(f"Failed to write full MILP result CSV: {full_log_path}")

    def _get_mean_capacity(self) -> float:
        """Return the time-averaged serving-cell capacity, zeroed during outage steps."""
        outage = (self._ho_exec_running + self._rlf_running) > 0.5
        cap_serving = self._data.capacity[self._pcells, np.arange(self._data.n_steps)]
        cap_serving = np.where(outage, 0.0, cap_serving)
        return float(np.sum(cap_serving)) / self._data.n_steps

    def _get_connected_time(self) -> int:
        """Return the number of steps where neither HO execution nor RLF recovery is active."""
        return int(
            self._data.n_steps
            - np.sum((self._ho_exec_running + self._rlf_running) > 0.5)
        )

    def _get_num_pp(self) -> tuple[int, float]:
        """Count ping-pong handovers in the solved assignment.

        Delegates to :func:`ho_optim_milp.common.metrics.count_ping_pongs`
        so the MILP and the standalone A3 reference apply
        one definition of ping-pong rather than three transcriptions of
        it.

        Returns
        -------
        tuple[int, float]
            ``(num_pp, pp_per_second)``
        """
        num_pp = count_ping_pongs(
            self._pcells,
            self._ho_exec_end > 0.5,
            t_mts_steps=self._rrc_consts.t_mts,
        )
        dt_s = self._data.n_steps * self._rrc_consts.t_res_ms / 1_000
        return num_pp, num_pp / dt_s

    def _get_snapshot(self, time_step: int) -> dict[str, Any]:
        """Return a per-time-step snapshot dict of all state variables and signal metrics."""
        if time_step < 0 or time_step >= self._data.n_steps:
            raise ValueError(
                f"time_step {time_step} is out of bounds [0, {self._data.n_steps})."
            )
        t = time_step
        return {
            "time_step": t,
            "ue_idx": self._data.ue_idx,
            "x_bs": self._x_bs[:, t].astype(int).tolist(),
            MILP_PCELL_COLUMN: int(self._pcells[t]),
            "i_q_in": int(self._i_q_in[t]),
            "i_q_out": int(self._i_q_out[t]),
            "u_n310_raw": int(self._n310_u_raw[t]),
            "u_n310_sat": int(self._n310_u_sat[t]),
            "c_n310": int(self._n310_cnt[t]),
            "i_n310": int(self._n310_ind[t]),
            "n310_b_reset": int(self._n310_b_reset[t]),
            "n310_b_q": int(self._n310_b_q[t]),
            "n310_b_c": int(self._n310_b_c[t]),
            "u_n311_raw": int(self._n311_u_raw[t]),
            "u_n311_sat": int(self._n311_u_sat[t]),
            "c_n311": int(self._n311_cnt[t]),
            "i_n311": int(self._n311_ind[t]),
            "n311_b_reset": int(self._n311_b_reset[t]),
            "n311_b_q": int(self._n311_b_q[t]),
            "n311_b_c": int(self._n311_b_c[t]),
            "i_t310_start": int(self._t310_start[t]),
            "i_t310_stop": int(self._t310_stop[t]),
            "i_t310_cancel_due_to_n311": int(self._t310_cancel_due_to_n311[t]),
            "i_t310_expiry": int(self._t310_expiry[t]),
            "tau_t310_rem": int(self._t310_tau[t]),
            "i_t310_eq_1": int(self._t310_tau_eq_1[t]),
            "i_t310_b_mode_rlf": int(self._t310_b_mode_rlf[t]),
            "i_t310_b_mode_start": int(self._t310_b_mode_start[t]),
            "i_t310_b_mode_stop": int(self._t310_b_mode_stop[t]),
            "i_t310_b_mode_decr": int(self._t310_b_mode_decr[t]),
            "i_rlf_start": int(self._rlf_start[t]),
            "i_rlf_in_last_t_ho_steps": int(self._rlf_in_last_t_ho_steps[t]),
            "i_ho_same_cell": self._ho_same_cell[:, t].astype(int).tolist(),
            "i_ho_cell_change": int(self._ho_cell_change[t]),
            "i_ho_exec_end": int(self._ho_exec_end[t]),
            "i_ho_exec_running": int(self._ho_exec_running[t]),
            "i_t310_active": int(self._t310_active[t]),
            "i_rlf_running": int(self._rlf_running[t]),
            "i_rate_conn_aux": int(self._rate_conn_aux[t]),
            "s_rate": float(self._rate_s_rate[t]),
            "r_rate": float(self._rate_r_rate[t]),
            "sinr_db": self._data.sinr_db[:, t].tolist(),
            "capacity": self._data.capacity[:, t].tolist(),
        }
