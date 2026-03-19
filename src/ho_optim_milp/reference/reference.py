"""RRC Simulation"""

import os
import csv
from typing import Any, TYPE_CHECKING

import numpy as np

from .core import VirtualClock
from .rrc import ReferenceRRC
from ..dataloader.preprocessing import preprocess_dataset

if TYPE_CHECKING:
    from .rrc_config import RRCConfig
    from ..result_manager.episode_result import EpisodeResult


class RRCReferenceSimulation:
    """RRC Reference Simulation (for testing and validation)."""

    sinr_db: np.ndarray

    def __init__(
        self,
        rrc_config: "RRCConfig",
        data: "EpisodeResult",
        lambda_r: float = 0.0,
        log_full_results: bool = False,
    ) -> None:
        self.rrc_config = rrc_config
        self.data = data

        self.lambda_r = lambda_r
        self.log_full_results_to_csv = log_full_results

        # Dimensions
        if data.rsrp.ndim == 3:
            self.n_steps, self.n_ue, self.n_bs = data.rsrp.shape
        else:
            self.n_ue = 1
            self.n_steps, self.n_bs = data.rsrp.shape

        self.tti_ms = 1  # 1 ms TTI
        self.simulation_time_ms = self.n_steps * rrc_config.t_res_ms

        self.phy_update_interval_ms = rrc_config.t_res_ms
        self.msr_interval_ms = rrc_config.t_res_ms

        self.clock = VirtualClock(
            tick_ms=self.tti_ms, simulation_time_ms=self.simulation_time_ms
        )
        self.ue_list = [
            ReferenceRRC(imsi=i, clock=self.clock, rrc_config=rrc_config)
            for i in range(self.n_ue)
        ]

        # Logging
        self.log_interval_ms = rrc_config.t_res_ms
        log_dir = os.path.join("results", "reference")
        self.full_log_path = os.path.join(log_dir, "reference.csv")
        os.makedirs(log_dir, exist_ok=True)

        self._result_buffer = {
            "pcell": [],
            "ho_exec_start": [],
            "ho_exec_end": [],
            "ho_exec_running": [],
            "hof_detected": [],
            "rlf_start": [],
            "rlf_running": [],
        }

    def run(self) -> None:
        """Run the RRC reference simulation."""
        # Preprocess dataset
        rsrp_dbm, sinr_db, q_in_mat, q_out_mat = preprocess_dataset(
            self.data, self.rrc_config, use_l3_filtering=True
        )
        self.sinr_db = sinr_db

        # Initial PHY update
        for i, ue in enumerate(self.ue_list):
            ue.receive_phy_update(
                rsrp_dbm[:, i, 0],
                sinr_db[:, i, 0],
                q_in_mat[:, i, 0],
                q_out_mat[:, i, 0],
            )
            ue.initial_access()

        # Main deterministic loop – each iteration = 1 ms (TTI)
        if self.log_full_results_to_csv:
            with open(self.full_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.ue_list[0].snapshot().keys(), delimiter=";"
                )
                writer.writeheader()

                # Simulation loop
                for _ in range(self.clock.n_steps):
                    # Step the simulation
                    self._simulation_step(rsrp_dbm, sinr_db, q_in_mat, q_out_mat)

                    # Log results to CSV
                    if (self.clock.now % self.log_interval_ms) == 0:
                        for ue in self.ue_list:
                            writer.writerow(ue.snapshot())

                    # Log snapshot
                    self._log_snapshot()

                    # 5. Advance clock
                    self.clock.advance(step_ms=self.tti_ms)
        else:
            # Simulation loop
            for _ in range(self.clock.n_steps):
                # Log snapshot
                self._log_snapshot()

                # Step the simulation
                self._simulation_step(rsrp_dbm, sinr_db, q_in_mat, q_out_mat)

                # 5. Advance clock
                self.clock.advance(step_ms=self.tti_ms)

        print("RRC UE simulation completed.")

    def _simulation_step(
        self,
        rsrp_dbm: np.ndarray,
        sinr_db: np.ndarray,
        q_in_mat: np.ndarray,
        q_out_mat: np.ndarray,
    ) -> None:
        # 1. PHY update
        if (self.clock.now % self.phy_update_interval_ms) == 0:
            idx = self.clock.now // self.phy_update_interval_ms
            for i, ue in enumerate(self.ue_list):
                ue.receive_phy_update(
                    rsrp_dbm[:, i, idx],
                    sinr_db[:, i, idx],
                    q_in_mat[:, i, idx],
                    q_out_mat[:, i, idx],
                )
                ue.update_sync_state()

        # 2. Process scheduled events
        self.clock.run_due_events()

        # 3. Measurement reporting and HO decisions
        if (self.clock.now % self.msr_interval_ms) == 0:
            for ue in self.ue_list:
                ue.process_measurements()

    def _log_snapshot(self) -> None:
        # Log snapshot
        if (self.clock.now % self.log_interval_ms) == 0:
            # Store results in buffer
            self._result_buffer["pcell"].append([ue.pcell for ue in self.ue_list])
            self._result_buffer["ho_exec_start"].append(
                [int(ue.flag_ho_exec_started) for ue in self.ue_list]
            )
            self._result_buffer["ho_exec_end"].append(
                [int(ue.flag_ho_exec_success) for ue in self.ue_list]
            )
            self._result_buffer["ho_exec_running"].append(
                [int(ue.ho_exec_ongoing) for ue in self.ue_list]
            )
            self._result_buffer["rlf_start"].append(
                [int(ue.flag_rlf_started) for ue in self.ue_list]
            )
            self._result_buffer["rlf_running"].append(
                [int(ue.rlfr_ongoing) for ue in self.ue_list]
            )
            self._result_buffer["hof_detected"].append(
                [int(ue.flag_hof_detected) for ue in self.ue_list]
            )

    def extract_results(self) -> dict[str, np.ndarray]:
        """Extract results from the simulation.

        Returns arrays shaped (n_logs, n_ue) for all keys in the buffer.
        """
        out: dict[str, np.ndarray] = {}
        for k, v in self._result_buffer.items():
            arr = np.asarray(v)
            if arr.ndim == 1:
                # In case something was logged as a scalar per log step, upgrade to (n_logs, 1)
                arr = arr[:, None]
            out[k] = arr
        return out

    def get_result_metrics(
        self, aggregated: bool = True
    ) -> dict[str, Any] | dict[int, dict[str, Any]]:
        """Compute and return result metrics.

        Parameters
        ----------
        aggregated : bool, default True
            - True: metrics aggregated over all UEs
            - False: dict keyed by ue index -> per-UE metrics dict
        """
        if aggregated:
            return self._get_result_metrics_aggregated()
        return self._get_result_metrics_per_ue()

    def _get_result_metrics_aggregated(self) -> dict[str, Any]:
        """Aggregated metrics over all UEs."""
        dt_s = self.simulation_time_ms / 1000.0
        n_total_samples = int(self.n_steps * self.n_ue)

        ref_c_mean, max_c_mean = self._get_mean_capacity(aggregated=True)
        ref_t_connected = self._get_connected_time(aggregated=True)

        ref_n_ho = int(np.sum(self._result_buffer["ho_exec_end"]))
        ref_num_pp, ref_pp_per_s = self._get_num_pp(aggregated=True)
        ref_num_hof = int(np.sum(self._result_buffer["hof_detected"]))
        ref_num_rlf = int(np.sum(self._result_buffer["rlf_start"]))

        return {
            "simulation_id": self.rrc_config.simulation_id,
            "simulated_time_s": float(dt_s),
            "simulated_steps": int(n_total_samples),
            "mean_capacity": float(ref_c_mean),
            "max_mean_capacity": float(max_c_mean),
            "rel_connected_time": float(ref_t_connected) / float(n_total_samples),
            "num_ho": ref_n_ho,
            "ho_per_s": float(ref_n_ho / dt_s),
            "num_pp": int(ref_num_pp),
            "pp_per_s": float(ref_pp_per_s),
            "pp_rate": (float(ref_num_pp / ref_n_ho) if ref_n_ho > 0 else float("inf")),
            "num_hof": ref_num_hof,
            "hof_rate": float(
                ref_num_hof / (ref_num_hof + ref_n_ho)
                if (ref_num_hof + ref_n_ho) > 0
                else 0.0
            ),
            "num_rlf": ref_num_rlf,
        }

    def _get_result_metrics_per_ue(self) -> dict[int, dict[str, Any]]:
        """Per-UE metrics (dict keyed by UE index)."""
        dt_s = self.simulation_time_ms / 1000.0

        ref_c_mean_ue, max_c_mean_ue = self._get_mean_capacity(aggregated=False)
        ref_t_connected_ue = np.asarray(
            self._get_connected_time(aggregated=False)
        )  # (n_ue,)

        ho_end = np.asarray(self._result_buffer["ho_exec_end"])  # (n_steps, n_ue)
        hof = np.asarray(self._result_buffer["hof_detected"])  # (n_steps, n_ue)
        rlf = np.asarray(self._result_buffer["rlf_start"])  # (n_steps, n_ue)

        if ho_end.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"ho_exec_end has shape {ho_end.shape}, expected {(self.n_steps, self.n_ue)}"
            )
        if hof.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"hof_detected has shape {hof.shape}, expected {(self.n_steps, self.n_ue)}"
            )
        if rlf.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"rlf_start has shape {rlf.shape}, expected {(self.n_steps, self.n_ue)}"
            )

        pp_counts_ue, pp_per_s_ue = self._get_num_pp(aggregated=False)

        out: dict[int, dict[str, Any]] = {}
        for ue_idx in range(self.n_ue):
            ref_n_ho = int(np.sum(ho_end[:, ue_idx]))
            ref_num_pp = int(np.asarray(pp_counts_ue)[ue_idx])
            ref_num_hof = int(np.sum(hof[:, ue_idx]))

            out[int(ue_idx)] = {
                "simulation_id": self.rrc_config.simulation_id,
                "simulated_time_s": float(dt_s),
                "simulated_steps": self.n_steps,
                "ue_index": ue_idx,
                "ref_mean_capacity": float(np.asarray(ref_c_mean_ue)[ue_idx]),
                "max_mean_capacity": float(np.asarray(max_c_mean_ue)[ue_idx]),
                "ref_rel_connected_time": float(ref_t_connected_ue[ue_idx])
                / float(self.n_steps),
                "ref_num_ho": ref_n_ho,
                "ref_ho_per_s": ref_n_ho / dt_s,
                "ref_num_pp": ref_num_pp,
                "ref_pp_per_s": float(np.asarray(pp_per_s_ue)[ue_idx]),
                "ref_pp_rate": (
                    (ref_num_pp / ref_n_ho) if ref_n_ho > 0 else float("inf")
                ),
                "ref_num_hof": ref_num_hof,
                "ref_hof_rate": (
                    ref_num_hof / (ref_num_hof + ref_n_ho)
                    if (ref_num_hof + ref_n_ho) > 0
                    else 0.0
                ),
                "ref_num_rlf": int(np.sum(rlf[:, ue_idx])),
            }

        return out

    def _get_achieved_capacity(self) -> np.ndarray:
        """Achieved capacity of each UE."""
        if self.sinr_db.ndim != 3:
            raise ValueError(f"Expected sinr_db.ndim == 3, got {self.sinr_db.ndim}")

        n_bs, n_ue, n_steps = self.sinr_db.shape
        if n_ue != self.n_ue or n_steps != self.n_steps:
            raise ValueError(
                "sinr_db shape mismatch. "
                f"sinr_db={(n_bs, n_ue, n_steps)}, expected (*, {self.n_ue}, {self.n_steps})."
            )

        # Capacity matrix [bs, ue, t]
        capacity_mat = np.log2(1.0 + 10.0 ** (self.sinr_db / 10.0))

        pcells = np.asarray(self._result_buffer["pcell"])  # (n_steps, n_ue)
        ho_run = np.asarray(self._result_buffer["ho_exec_running"])  # (n_steps, n_ue)
        rlf_run = np.asarray(self._result_buffer["rlf_running"])  # (n_steps, n_ue)

        if pcells.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"pcell has shape {pcells.shape}, expected {(self.n_steps, self.n_ue)}"
            )
        if ho_run.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"ho_exec_running has shape {ho_run.shape}, expected {(self.n_steps, self.n_ue)}"
            )
        if rlf_run.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"rlf_running has shape {rlf_run.shape}, expected {(self.n_steps, self.n_ue)}"
            )

        # Disconnected if HO execution or RLF is running
        outage = (ho_run == 1) | (rlf_run == 1)  # (n_steps, n_ue)

        # cap_serving[t, ue] = capacity_mat[pcell[t, ue], ue, t]
        cap_serving = capacity_mat[
            pcells.T, np.arange(self.n_ue)[:, None], np.arange(self.n_steps)[None, :]
        ].T  # (n_steps, n_ue)

        # Set capacity to 0 during outage
        cap_serving[outage] = 0.0
        cap_achieved = cap_serving.T

        return cap_achieved  # (n_ue, n_steps)

    def _get_mean_capacity(
        self, aggregated: bool
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Mean capacity of serving cell (reference) and maximum mean capacity.

        Parameters
        ----------
        aggregated : bool, default True
            - True: returns (ref_c_mean, max_c_mean) as floats
            - False: returns (ref_c_mean_ue, max_c_mean_ue) as arrays of shape (n_ue,)
        """
        # Capacity matrix [bs, ue, t]
        capacity_mat = np.log2(1.0 + 10.0 ** (self.sinr_db / 10.0))

        cap_achieved = self._get_achieved_capacity()  # (n_ue, n_steps)

        if aggregated:
            ref_c_sum = float(np.sum(cap_achieved))
            ref_c_mean = ref_c_sum / float(self.n_steps * self.n_ue)
            max_c_mean = float(
                np.mean(np.max(capacity_mat, axis=0))
            )  # mean over (ue,t)
            return float(ref_c_mean), float(max_c_mean)

        # per UE
        ref_c_mean_ue = np.zeros(self.n_ue, dtype=np.float64)
        for ue_idx in range(self.n_ue):
            ref_c_mean_ue[ue_idx] = float(
                np.sum(cap_achieved[ue_idx, :]) / self.n_steps
            )

        max_c_mean_ue = np.mean(np.max(capacity_mat, axis=0), axis=1).astype(
            np.float64, copy=False
        )
        return ref_c_mean_ue, max_c_mean_ue

    def _get_connected_time(self, aggregated: bool) -> int | np.ndarray:
        """Connected time in samples.

        Parameters
        ----------
        aggregated : bool, default True
            - True: total connected samples across all UEs (int)
            - False: connected samples per UE (array shape (n_ue,))
        """
        ho_run = np.asarray(self._result_buffer["ho_exec_running"])  # (n_steps, n_ue)
        rlf_run = np.asarray(self._result_buffer["rlf_running"])  # (n_steps, n_ue)

        if ho_run.shape != (self.n_steps, self.n_ue) or rlf_run.shape != (
            self.n_steps,
            self.n_ue,
        ):
            raise ValueError(
                "Result buffers have unexpected shapes: "
                f"ho_exec_running={ho_run.shape}, rlf_running={rlf_run.shape}, "
                f"expected {(self.n_steps, self.n_ue)}."
            )

        disconnected = (ho_run + rlf_run) > 0.5  # (n_steps, n_ue)

        if aggregated:
            return int(self.n_steps * self.n_ue - int(np.sum(disconnected)))

        # Return per-UE connected time as array
        connected_per_ue = self.n_steps - np.sum(disconnected, axis=0)
        return connected_per_ue.astype(np.int64, copy=False)

    def _get_num_pp(
        self, aggregated: bool
    ) -> tuple[int, float] | tuple[np.ndarray, np.ndarray]:
        """Ping-pong count.

        Parameters
        ----------
        aggregated : bool, default True
            - True: (total_pp, total_pp_per_s)
            - False: (pp_counts_ue, pp_per_s_ue), each shape (n_ue,)
        """
        pcells = np.asarray(self._result_buffer["pcell"])  # (n_steps, n_ue)
        ho_end = np.asarray(self._result_buffer["ho_exec_end"])  # (n_steps, n_ue)

        if pcells.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"pcell has shape {pcells.shape}, expected {(self.n_steps, self.n_ue)}"
            )
        if ho_end.shape != (self.n_steps, self.n_ue):
            raise ValueError(
                f"ho_exec_end has shape {ho_end.shape}, expected {(self.n_steps, self.n_ue)}"
            )

        t_mts_ms = self.ue_list[0].t_mts_ms
        t_mts_samples = int(t_mts_ms // self.phy_update_interval_ms)

        pp_counts = np.zeros(self.n_ue, dtype=np.int64)

        for ue_idx in range(self.n_ue):
            ho_end_idx = np.where(ho_end[:, ue_idx] > 0.5)[0]
            if ho_end_idx.size < 2:
                continue

            for t_ho1, t_ho2 in zip(ho_end_idx[:-1], ho_end_idx[1:]):
                if (t_ho2 - t_ho1) <= t_mts_samples:
                    if t_ho1 > 0 and pcells[t_ho1 - 1, ue_idx] == pcells[t_ho2, ue_idx]:
                        pp_counts[ue_idx] += 1

        dt_s = self.simulation_time_ms / 1000.0
        pp_per_s = pp_counts.astype(np.float64) / float(dt_s)

        if aggregated:
            total_pp = int(np.sum(pp_counts))
            total_pp_per_s = float(total_pp / dt_s)
            return total_pp, total_pp_per_s

        return pp_counts, pp_per_s

    def _compute_objective_value(self, aggregated: bool) -> float | np.ndarray:
        """Compute an objective value for optimization purposes."""
        cap_achieved = self._get_achieved_capacity()  # (n_ue, n_steps)
        mean_cap_ue = np.mean(cap_achieved, axis=1)  # (n_ue,)

        ho_run = np.asarray(self._result_buffer["ho_exec_running"])  # (n_steps, n_ue)
        rlf_run = np.asarray(self._result_buffer["rlf_running"])  # (n_steps, n_ue)

        if aggregated:
            obj_rate = np.mean(cap_achieved)
            obj_out = np.mean(ho_run + rlf_run)
            return float(obj_rate - self.lambda_r * obj_out)

        obj_values = np.zeros(self.n_ue, dtype=np.float64)
        for ue_idx in range(self.n_ue):
            obj_rate = mean_cap_ue[ue_idx]
            obj_out = (
                np.sum(ho_run[:, ue_idx]) + np.sum(rlf_run[:, ue_idx])
            ) / self.n_steps
            obj_values[ue_idx] = obj_rate - self.lambda_r * obj_out
        return obj_values
