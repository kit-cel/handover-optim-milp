"""Result extraction and reporting for a solved MILP handover optimizer."""

import csv
import os
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .milp_variables import ModelVars
    from .milp import ProblemData, RRCConstants


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
        self._pcells: np.ndarray = np.argmax(v.x_bs.X, axis=0)
        self._i_q_in: np.ndarray = np.asarray(v.i_q_in.X)
        self._i_q_out: np.ndarray = np.asarray(v.i_q_out.X)
        self._n310_u_raw: np.ndarray = np.asarray(v.n310.u_raw.X)
        self._n310_cnt: np.ndarray = np.asarray(v.n310.cnt.X)
        self._n311_u_raw: np.ndarray = np.asarray(v.n311.u_raw.X)
        self._n311_cnt: np.ndarray = np.asarray(v.n311.cnt.X)
        self._t310_start: np.ndarray = np.asarray(v.t310.start.X)
        self._t310_stop: np.ndarray = np.asarray(v.t310.stop.X)
        self._t310_tau: np.ndarray = np.asarray(v.t310.tau.X)
        self._t310_tau_eq_1: np.ndarray = np.asarray(v.t310.tau_eq_1.X)
        self._t310_active: np.ndarray = np.asarray(v.t310.active.X)
        self._rlf_start: np.ndarray = np.asarray(v.rlf.start.X)
        self._rlf_running: np.ndarray = np.asarray(v.rlf.running.X)
        self._ho_exec_end: np.ndarray = np.asarray(v.ho.exec_end.X)
        self._ho_exec_running: np.ndarray = np.asarray(v.ho.exec_running.X)

    def extract_results(self) -> dict[str, np.ndarray]:
        """Return a flat dict of all key solution arrays."""
        return {
            "sinr_db": self._data.sinr_db,
            "capacity": self._data.capacity,
            "pcell": self._pcells,
            "i_q_in": self._i_q_in,
            "i_q_out": self._i_q_out,
            "u_n310_raw": self._n310_u_raw,
            "c_n310": self._n310_cnt,
            "u_n311_raw": self._n311_u_raw,
            "c_n311": self._n311_cnt,
            "i_t310_start": self._t310_start,
            "i_t310_stop": self._t310_stop,
            "tau_t310_rem": self._t310_tau,
            "i_rlf_start": self._rlf_start,
            "i_ho_exec_end": self._ho_exec_end,
            "i_ho_exec_running": self._ho_exec_running,
            "i_t310_active": self._t310_active,
            "i_rlf_running": self._rlf_running,
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

        out: dict[int, dict[str, Any]] = {}
        out[int(self._data.ue_idx)] = {
            "simulation_id": self._simulation_id,
            "ue_idx": self._data.ue_idx,
            "simulated_time_s": dt_s,
            "simulated_steps": self._data.n_steps,
            "mean_capacity": c_mean,
            "max_mean_capacity": float(np.mean(np.max(self._data.capacity, axis=0))),
            "rel_connected_time": t_connected / self._data.n_steps,
            "num_ho": num_ho,
            "ho_per_s": num_ho / dt_s,
            "num_pp": num_pp,
            "pp_per_s": pp_per_s,
            "pp_rate": num_pp / num_ho if num_ho > 0 else float("inf"),
            "num_rlf": int(np.sum(self._rlf_start)),
            "ho_objective_value": obj_value,
        }
        return out

    def log_results_to_csv(
        self, file_name: str | None = None, subfolder: str | None = None
    ) -> None:
        """Write a per-step snapshot to a CSV file (append mode).

        Parameters
        ----------
        file_name : str, optional
            Output file name.  Auto-generated from simulation/episode/UE
            metadata when omitted.
        subfolder : str, optional
            Subdirectory inside ``log_dir`` for the file.
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
        file_exists = os.path.exists(full_log_path)

        with open(full_log_path, "a", newline="", encoding="utf-8") as f:
            header = self._get_snapshot(0).keys()
            writer = csv.DictWriter(f, fieldnames=header, delimiter=";")
            if not file_exists:
                writer.writeheader()
            for i in range(self._data.n_steps):
                writer.writerow(self._get_snapshot(i))

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
        """Count ping-pong handovers: consecutive HO completions within ``t_mts``
        steps that return to the previous PCell.

        Returns
        -------
        tuple[int, float]
            ``(num_pp, pp_per_second)``
        """
        t_ho_end = np.where(self._ho_exec_end > 0.5)[0]
        num_pp = 0
        for t_ho1, t_ho2 in zip(t_ho_end[:-1], t_ho_end[1:]):
            if t_ho2 - t_ho1 <= self._rrc_consts.t_mts:
                if self._pcells[t_ho1 - 1] == self._pcells[t_ho2]:
                    num_pp += 1
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
            "pcell": int(self._pcells[t]),
            "i_q_in": int(self._i_q_in[t]),
            "i_q_out": int(self._i_q_out[t]),
            "u_n310_raw": int(self._n310_u_raw[t]),
            "c_n310": int(self._n310_cnt[t]),
            "u_n311_raw": int(self._n311_u_raw[t]),
            "c_n311": int(self._n311_cnt[t]),
            "i_t310_start": int(self._t310_start[t]),
            "i_t310_stop": int(self._t310_stop[t]),
            "tau_t310_rem": int(self._t310_tau[t]),
            "i_t310_eq_1": int(self._t310_tau_eq_1[t]),
            "i_rlf_start": int(self._rlf_start[t]),
            "i_ho_exec_end": int(self._ho_exec_end[t]),
            "i_ho_exec_running": int(self._ho_exec_running[t]),
            "i_t310_active": int(self._t310_active[t]),
            "i_rlf_running": int(self._rlf_running[t]),
            "sinr_db": self._data.sinr_db[:, t].tolist(),
            "capacity": self._data.capacity[:, t].tolist(),
        }
