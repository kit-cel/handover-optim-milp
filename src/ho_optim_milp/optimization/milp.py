"""
MILP optimizer implementation using Gurobi.

Optimizations implemented:
- Remove y_tau one-hot encoding for tau_t310 (exact).
- Remove i_connected[n,t] binary matrix; use per-time achieved rate r_t (exact).
- Optional guaranteed-safe candidate pruning as a warm-start (does not affect final optimality).
"""

from typing import Any, TYPE_CHECKING
import os
import csv
import multiprocessing

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from . import optim_utils as ut
from .base_optimizer import GurobiBaseOptimizer
from ..dataloader.preprocessing import preprocess_dataset
from ..reference.reference import RRCReferenceSimulation

if TYPE_CHECKING:
    from .optim_config import OptimConfig
    from ..result_manager.episode_result import EpisodeResult


class HandoverOptimizerMILP(GurobiBaseOptimizer):
    """Gurobi-based MILP handover optimizer."""

    debug = False

    def __init__(
        self,
        config: "OptimConfig",
        data: "EpisodeResult",
        ue_idx: int,
        *,
        use_safe_candidate_pruning: bool | None = None,
        candidate_k: int | None = None,
        candidate_delta_db: float | None = None,
        candidate_metric: str | None = None,
        candidate_include_prev: bool = True,
        candidate_include_forced: bool = True,
        safe_pruning_time_limit_s: float | None = None,
        safe_pruning_mip_gap: float | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.data = data
        self.ue_idx = ue_idx

        self._max_threads = multiprocessing.cpu_count()

        # Lagrange multiplier for HO failures and RLFs
        self.lambda_r = config.lambda_r

        # RRC Parameters
        self.t_res_ms = config.rrc.t_res_ms
        self.q_in_db_const = config.rrc.q_in_db
        self.q_out_db_const = config.rrc.q_out_db
        self.n310_const = config.rrc.n310
        self.n311_const = config.rrc.n311

        self.t310_const = config.rrc.t310_ms // self.t_res_ms
        self.t_ho_prep_const = config.rrc.t_ho_prep_ms_simulated // self.t_res_ms
        self.t_ho_exec_const = config.rrc.t_ho_exec_ms_simulated // self.t_res_ms
        self.t_ho_const = self.t_ho_prep_const + self.t_ho_exec_const
        self.t_rlfr_const = config.rrc.t_rlfr_ms_simulated // self.t_res_ms
        self.t_mts_const = config.rrc.t_mts_ms // self.t_res_ms

        # Logging
        self.log_dir = os.path.join("results", "optimization")
        os.makedirs(self.log_dir, exist_ok=True)

        # Reference result buffer
        self.ref_results: dict[str, np.ndarray] | None = None

        # Safe candidate pruning options
        self.use_safe_candidate_pruning = bool(
            use_safe_candidate_pruning
            if use_safe_candidate_pruning is not None
            else getattr(config, "use_safe_candidate_pruning", False)
        )
        self.candidate_k = int(
            candidate_k
            if candidate_k is not None
            else getattr(config, "candidate_k", 0)
        )
        self.candidate_delta_db = (
            float(candidate_delta_db)
            if candidate_delta_db is not None
            else float(getattr(config, "candidate_delta_db", 0.0))
        )
        self.candidate_metric = (
            str(candidate_metric)
            if candidate_metric is not None
            else str(getattr(config, "candidate_metric", "rsrp"))
        )
        self.candidate_include_prev = bool(candidate_include_prev)
        self.candidate_include_forced = bool(candidate_include_forced)
        self.safe_pruning_time_limit_s = (
            float(safe_pruning_time_limit_s)
            if safe_pruning_time_limit_s is not None
            else float(getattr(config, "safe_pruning_time_limit_s", 0.0))
        )
        self.safe_pruning_mip_gap = (
            float(safe_pruning_mip_gap)
            if safe_pruning_mip_gap is not None
            else float(getattr(config, "safe_pruning_mip_gap", 0.0))
        )

        # Data
        self.load_data(data)

        # Optional safe pruning warm-start
        self._warm_start_x: np.ndarray | None = None
        if self.use_safe_candidate_pruning and (
            self.candidate_k > 0 or self.candidate_delta_db > 0.0
        ):
            self._warm_start_x = self._solve_pruned_for_warm_start()

        # Build final model (full, unpruned -> optimality preserved)
        self.setup_model(name="HandoverOptimizerMILP")
        self.model.setParam("Threads", self._max_threads)

        self._add_variables()
        self.model.update()

        self._add_constraints()
        self._set_objective()

        # Apply warm start (safe pruning warm start or others)
        if self._warm_start_x is not None:
            self._apply_warm_start(self._warm_start_x)

    def load_data(self, data: "EpisodeResult") -> None:
        """Data loading."""
        self.rsrp_dbm, self.sinr_db, self.q_in_mat, self.q_out_mat = preprocess_dataset(
            self.data, self.rrc_config, use_l3_filtering=True, ue_no=self.ue_idx
        )

        self.n_bs, self.n_steps = self.sinr_db.shape

        if self.debug:
            sim = RRCReferenceSimulation(self.config.rrc, data)
            sim.run()
            self.ref_results = sim.extract_results()

    def _compute_candidate_sets(self) -> list[np.ndarray]:
        """Candidate pruning (safe warm start)"""
        metric = self.candidate_metric.lower()
        if metric == "sinr":
            score = self.sinr_db
        else:
            score = self.rsrp_dbm

        candidates: list[np.ndarray] = []
        prev_cand: set[int] = set()

        for t in range(self.n_steps):
            s = score[:, t]
            idx = np.arange(self.n_bs)

            cand_set: set[int] = set()
            if self.candidate_k > 0:
                k = min(self.candidate_k, self.n_bs)
                topk = idx[np.argpartition(-s, k - 1)[:k]]
                cand_set.update(int(i) for i in topk)

            if self.candidate_delta_db > 0.0:
                best = float(np.max(s))
                mask = s >= (best - self.candidate_delta_db)
                cand_set.update(int(i) for i in idx[mask])

            if not cand_set:
                cand_set.add(int(np.argmax(s)))

            if self.candidate_include_prev and prev_cand:
                cand_set.update(prev_cand)

            candidates.append(np.array(sorted(cand_set), dtype=int))
            prev_cand = cand_set

        return candidates

    def _solve_pruned_for_warm_start(self) -> np.ndarray | None:
        candidates = self._compute_candidate_sets()

        pruned = gp.Model("HandoverOptimizerMILP_pruned")
        pruned.setParam("Threads", self._max_threads)
        pruned.setParam("OutputFlag", 0)

        if self.safe_pruning_time_limit_s and self.safe_pruning_time_limit_s > 0:
            pruned.setParam("TimeLimit", self.safe_pruning_time_limit_s)
        if self.safe_pruning_mip_gap and self.safe_pruning_mip_gap > 0:
            pruned.setParam("MIPGap", self.safe_pruning_mip_gap)

        ub_x = np.zeros((self.n_bs, self.n_steps), dtype=float)
        for t, cand in enumerate(candidates):
            ub_x[cand, t] = 1.0

        x_bs = pruned.addMVar(
            (self.n_bs, self.n_steps), vtype=GRB.BINARY, ub=ub_x, name="x_bs"
        )

        # Quick, logic-free warm start objective: maximize average capacity under
        # connectivity proxy. This does not alter final optimality because it is
        # only for warm start.
        cap = np.log2(1 + 10 ** (self.sinr_db / 10))
        pruned.addConstr(x_bs.sum(axis=0) == 1, name="onehot")
        obj = (cap * x_bs).sum() / self.n_steps
        pruned.setObjective(obj, GRB.MAXIMIZE)

        try:
            pruned.optimize()
        except gp.GurobiError:
            return None

        if pruned.SolCount <= 0:
            return None

        x_val = x_bs.X
        # Ensure exact one-hot by argmax in case of numerical tolerances
        pcells = np.argmax(x_val, axis=0)
        warm = np.zeros_like(x_val, dtype=float)
        warm[pcells, np.arange(self.n_steps)] = 1.0
        return warm

    def _apply_warm_start(self, x_start: np.ndarray) -> None:
        if x_start.shape != (self.n_bs, self.n_steps):
            raise ValueError(
                f"x_start shape {x_start.shape} does not match ({self.n_bs}, {self.n_steps})"
            )

        for n in range(self.n_bs):
            for t in range(self.n_steps):
                self.x_bs[n, t].Start = float(x_start[n, t])

    def _add_variables(self) -> None:
        self.x_bs = self.model.addMVar(
            (self.n_bs, self.n_steps), vtype=GRB.BINARY, name="x_bs"
        )

        self.i_q_in = self.model.addMVar(self.n_steps, vtype=GRB.BINARY, name="i_q_in")
        self.i_q_out = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_q_out"
        )

        # N310
        self.u_n310_raw = self.model.addMVar(
            self.n_steps,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.n310_const + 1,
            name="u_n310_raw",
        )
        self.u_n310_sat = self.model.addMVar(
            self.n_steps, vtype=GRB.INTEGER, lb=0, ub=self.n310_const, name="u_n310_sat"
        )
        self.c_n310 = self.model.addMVar(
            self.n_steps, vtype=GRB.INTEGER, lb=0, ub=self.n310_const, name="c_n310"
        )
        self.i_n310 = self.model.addMVar(self.n_steps, vtype=GRB.BINARY, name="i_n310")
        self._b_reset_310 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_reset_310"
        )
        self._b_q_310 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_q_310"
        )
        self._b_c_310 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_c_310"
        )

        # N311
        self.u_n311_raw = self.model.addMVar(
            self.n_steps,
            vtype=GRB.INTEGER,
            lb=0,
            ub=self.n311_const + 1,
            name="u_n311_raw",
        )
        self.u_n311_sat = self.model.addMVar(
            self.n_steps, vtype=GRB.INTEGER, lb=0, ub=self.n311_const, name="u_n311_sat"
        )
        self.c_n311 = self.model.addMVar(
            self.n_steps, vtype=GRB.INTEGER, lb=0, ub=self.n311_const, name="c_n311"
        )
        self.i_n311 = self.model.addMVar(self.n_steps, vtype=GRB.BINARY, name="i_n311")
        self._b_reset_311 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_reset_311"
        )
        self._b_q_311 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_q_311"
        )
        self._b_c_311 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_c_311"
        )

        # T310 indicators
        self.i_t310_start = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_t310_start"
        )
        self.i_t310_stop = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_t310_stop"
        )
        self.i_t310_active = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_t310_active"
        )
        self._i_t310_cancel_due_to_n311 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_i_t310_cancel_due_to_n311"
        )
        self._i_t310_expiry = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_i_t310_expiry"
        )

        # T310/RLF auxiliary mode flags
        self._m_rlf = self.model.addMVar(self.n_steps, vtype=GRB.BINARY, name="m_rlf")
        self._m_start = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="m_start"
        )
        self._m_stop = self.model.addMVar(self.n_steps, vtype=GRB.BINARY, name="m_stop")
        self._m_decr = self.model.addMVar(self.n_steps, vtype=GRB.BINARY, name="m_decr")

        # T310 timer (integer only; no one-hot)
        self.tau_t310 = self.model.addMVar(
            self.n_steps, vtype=GRB.INTEGER, lb=0, ub=self.t310_const, name="tau_t310"
        )
        self._i_t310_eq_1 = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="tau_eq_1"
        )

        # RLF
        self.i_rlf_start = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_rlf_start"
        )
        self.i_rlf_running = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_rlf_running"
        )

        self._i_rlf_in_last_t_ho_steps = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_rlf_in_last_t_ho_steps"
        )

        # PCell change
        self.i_same_cell = self.model.addMVar(
            (self.n_bs, self.n_steps), vtype=GRB.BINARY, name="i_same_cell"
        )
        self.i_pcell_change = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_pcell_change"
        )

        # HO execution
        self.i_ho_exec_end = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_ho_exec_end"
        )
        self.i_ho_exec_running = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="i_ho_exec_running"
        )

        # Connectivity (per time)
        self._b_conn_aux = self.model.addMVar(
            self.n_steps, vtype=GRB.BINARY, name="_b_conn_aux"
        )

        # Rate aggregation (replaces i_connected[n,t])
        self.s_rate = self.model.addMVar(
            self.n_steps, vtype=GRB.CONTINUOUS, lb=0.0, name="s_rate"
        )
        self.r_rate = self.model.addMVar(
            self.n_steps, vtype=GRB.CONTINUOUS, lb=0.0, name="r_rate"
        )

        self._variables_added = True

    def _add_constraints(self) -> None:
        if not self._variables_added:
            raise RuntimeError("Variables must be added before adding constraints.")

        if self.debug:
            if self.ref_results is None:
                raise RuntimeError("Reference results not available for debugging.")
            ref_pcells = self.ref_results["pcell"]
            self.model.addConstr(
                self.x_bs == np.eye(self.n_bs, dtype=int)[:, ref_pcells], name="fix_x"
            )

        # one-hot PCell
        self.model.addConstr(self.x_bs.sum(axis=0) == 1, name="pcell_onehot")

        # q indicators
        for t in range(self.n_steps):
            self.model.addConstr(
                self.i_q_in[t] == (self.q_in_mat[:, t] * self.x_bs[:, t]).sum(),
                name=f"q_in_def_{t}",
            )
            self.model.addConstr(
                self.i_q_out[t] == (self.q_out_mat[:, t] * self.x_bs[:, t]).sum(),
                name=f"q_out_def_{t}",
            )

        # T310: define active and eq1 from integer tau (exact)
        # active: tau >= 1  <=>  tau >= i_active and tau <= t310*i_active
        for t in range(self.n_steps):
            self.model.addConstr(
                self.tau_t310[t] >= self.i_t310_active[t], name=f"t310_act_lb_{t}"
            )
            self.model.addConstr(
                self.tau_t310[t] <= self.t310_const * self.i_t310_active[t],
                name=f"t310_act_ub_{t}",
            )

            # eq1: z=1 <=> tau == 1
            z = self._i_t310_eq_1[t]
            self.model.addConstr(
                self.tau_t310[t] - 1 <= (self.t310_const - 1) * (1 - z),
                name=f"t310_eq1_ub_{t}",
            )
            self.model.addConstr(
                1 - self.tau_t310[t] <= 1 * (1 - z),
                name=f"t310_eq1_lb_{t}",
            )
            # tau = 1 => z = 1  (uses 'active' to eliminate the tau=0 case)
            # Equivalent to: if a=1 and z=0 then tau>=2
            if self.t310_const >= 2:
                self.model.addConstr(
                    self.tau_t310[t] >= 2 * (self.i_t310_active[t] - z),
                    name=f"t310_eq1_rev_{t}",
                )

        # RLF start (linear AND)
        self.model.addConstr(self.i_rlf_start[0] == 0, name="rlf_start_0")
        for t in range(1, self.n_steps):
            if self.debug:
                if self.ref_results is None:
                    raise RuntimeError("Reference results not available for debugging.")
                ref_hof = self.ref_results["hof_detected"]
                if ref_hof[t]:
                    self.model.addConstr(
                        self.i_rlf_start[t] == 1, name=f"rlf_force_{t}"
                    )
                    continue

            self.model.addConstr(
                self.i_rlf_start[t] <= self._i_t310_eq_1[t - 1],
                name=f"rlf_start_ub1_{t}",
            )
            self.model.addConstr(
                self.i_rlf_start[t] <= 1 - self.i_rlf_running[t - 1],
                name=f"rlf_start_ub2_{t}",
            )
            self.model.addConstr(
                self.i_rlf_start[t]
                >= self._i_t310_eq_1[t - 1] - self.i_rlf_running[t - 1],
                name=f"rlf_start_lb_{t}",
            )

        # RLF running: OR of starts in last t_rlfr steps (preserving your "can be longer" behavior)
        self.model.addConstr(self.i_rlf_running[0] == 0, name="rlf_run_0")
        for t in range(1, self.n_steps):
            tau_max = min(t, self.t_rlfr_const - 1)
            sum_starts = self.i_rlf_start[t - tau_max : t + 1].sum()

            # Avoid division: (tau_max+1) * rlf_running >= sum_starts
            self.model.addConstr(
                (tau_max + 1) * self.i_rlf_running[t] >= sum_starts,
                name=f"rlf_run_lb_{t}",
            )
            self.model.addConstr(
                self.i_rlf_running[t] <= sum_starts + self.i_rlf_running[t - 1],
                name=f"rlf_run_ub_{t}",
            )

        # i_rlf_in_last_t_ho_steps: any rlf_running within last t_ho_const steps (including t)
        for t in range(self.n_steps):
            tau_max = min(t, self.t_ho_const)
            sum_rlfr = self.i_rlf_running[t - tau_max : t + 1].sum()
            self.model.addConstr(
                sum_rlfr >= self._i_rlf_in_last_t_ho_steps[t], name=f"rlf_prev_lb_{t}"
            )
            self.model.addConstr(
                sum_rlfr <= (tau_max + 1) * self._i_rlf_in_last_t_ho_steps[t],
                name=f"rlf_prev_ub_{t}",
            )

        # N310
        for t in range(self.n_steps):
            self.model.addConstr(
                self._b_reset_310[t] <= 1 - self.i_rlf_running[t],
                name=f"reset310_ub1_{t}",
            )
            self.model.addConstr(
                self._b_reset_310[t] <= 1 - self.i_ho_exec_running[t],
                name=f"reset310_ub2_{t}",
            )
            self.model.addConstr(
                self._b_reset_310[t]
                >= 1 - self.i_rlf_running[t] - self.i_ho_exec_running[t],
                name=f"reset310_lb_{t}",
            )

            self.model.addConstr(
                self._b_c_310[t] <= self._b_reset_310[t], name=f"c310_ub1_{t}"
            )
            self.model.addConstr(
                self._b_c_310[t] <= 1 - self.i_t310_active[t], name=f"c310_ub2_{t}"
            )
            self.model.addConstr(
                self._b_c_310[t] >= self._b_reset_310[t] - self.i_t310_active[t],
                name=f"c310_lb_{t}",
            )

        self.model.addConstr(self._b_q_310[0] == self.i_q_out[0], name="bq310_0")
        for t in range(1, self.n_steps):
            self.model.addConstr(
                self._b_q_310[t] <= self.i_q_out[t], name=f"bq310_ub1_{t}"
            )
            self.model.addConstr(
                self._b_q_310[t] <= 1 - self.i_t310_active[t - 1], name=f"bq310_ub2_{t}"
            )
            self.model.addConstr(
                self._b_q_310[t] >= self.i_q_out[t] - self.i_t310_active[t - 1],
                name=f"bq310_lb_{t}",
            )

        self.model.addConstr(self.u_n310_raw[0] == self.i_q_out[0], name="u310_0")

        for t in range(1, self.n_steps):
            c_tmp = self.model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=self.n310_const + 1, name=f"temp310_{t}"
            )
            self.model.addConstr(
                c_tmp == self.c_n310[t - 1] + self._b_q_310[t], name=f"temp310_def_{t}"
            )

            u_raw_aux = self.model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=self.n310_const + 1, name=f"u310_rawtmp_{t}"
            )
            ut.lin_prod_bin_int(
                self.model,
                u_raw_aux,
                self._b_reset_310[t],
                c_tmp,
                self.n310_const + 1,
                name=f"u310_prod_{t}",
            )
            self.model.addConstr(
                self.u_n310_raw[t] == u_raw_aux, name=f"u310_raw_def_{t}"
            )

        for t in range(self.n_steps):
            ut.lin_saturate_min(
                self.model,
                self.u_n310_sat[t],
                self.u_n310_raw[t],
                self.n310_const,
                self.n310_const + 1,
                name=f"u310_sat_{t}",
            )

        for t in range(self.n_steps):
            c_tmp = self.model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=self.n310_const, name=f"c310_tmp_{t}"
            )
            ut.lin_prod_bin_int(
                self.model,
                c_tmp,
                self._b_c_310[t],
                self.u_n310_sat[t],
                self.n310_const,
                name=f"c310_prod_{t}",
            )
            self.model.addConstr(self.c_n310[t] == c_tmp, name=f"c310_def_{t}")

        ub_u_n310 = self.n310_const + 1
        for t in range(self.n_steps):
            self.model.addConstr(
                self.u_n310_raw[t] >= self.n310_const * self.i_n310[t],
                name=f"i310_lb_{t}",
            )
            self.model.addConstr(
                self.u_n310_raw[t]
                <= (self.n310_const - 1)
                + (ub_u_n310 - (self.n310_const - 1)) * self.i_n310[t],
                name=f"i310_ub_{t}",
            )

        # N311
        for t in range(self.n_steps):
            self.model.addConstr(
                self._b_reset_311[t] <= 1 - self.i_rlf_running[t],
                name=f"reset311_ub1_{t}",
            )
            self.model.addConstr(
                self._b_reset_311[t] <= 1 - self.i_ho_exec_running[t],
                name=f"reset311_ub2_{t}",
            )
            self.model.addConstr(
                self._b_reset_311[t]
                >= 1 - self.i_rlf_running[t] - self.i_ho_exec_running[t],
                name=f"reset311_lb_{t}",
            )

        self.model.addConstr(self._b_q_311[0] == 0, name="bq311_0")
        for t in range(1, self.n_steps):
            self.model.addConstr(
                self._b_q_311[t] <= self.i_q_in[t], name=f"bq311_ub1_{t}"
            )
            self.model.addConstr(
                self._b_q_311[t] <= self.i_t310_active[t - 1], name=f"bq311_ub2_{t}"
            )
            self.model.addConstr(
                self._b_q_311[t] >= self.i_q_in[t] + self.i_t310_active[t - 1] - 1,
                name=f"bq311_lb_{t}",
            )

        self.model.addConstr(self.u_n311_raw[0] == 0, name="u311_0")
        self.model.addConstr(self.c_n311[0] == 0, name="c311_0")

        for t in range(1, self.n_steps):
            c_tmp = self.model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=self.n311_const + 1, name=f"temp311_{t}"
            )
            self.model.addConstr(
                c_tmp == self.c_n311[t - 1] + self._b_q_311[t], name=f"temp311_def_{t}"
            )

            u_raw_aux = self.model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=self.n311_const + 1, name=f"u311_rawtmp_{t}"
            )
            ut.lin_prod_bin_int(
                self.model,
                u_raw_aux,
                self._b_reset_311[t],
                c_tmp,
                self.n311_const + 1,
                name=f"u311_prod_{t}",
            )
            self.model.addConstr(
                self.u_n311_raw[t] == u_raw_aux, name=f"u311_raw_def_{t}"
            )

        for t in range(self.n_steps):
            ut.lin_saturate_min(
                self.model,
                self.u_n311_sat[t],
                self.u_n311_raw[t],
                self.n311_const,
                self.n311_const + 1,
                name=f"u311_sat_{t}",
            )

        # b_c_311[t] = b_reset_311[t] AND (1 - i_t310_stop[t]) AND i_t310_active[t]
        for t in range(self.n_steps):
            aux1 = self.model.addVar(vtype=GRB.BINARY, name=f"aux311_{t}")
            self.model.addConstr(aux1 <= self._b_reset_311[t], name=f"aux311_ub1_{t}")
            self.model.addConstr(
                aux1 <= 1 - self.i_t310_stop[t], name=f"aux311_ub2_{t}"
            )
            self.model.addConstr(
                aux1 >= self._b_reset_311[t] - self.i_t310_stop[t],
                name=f"aux311_lb_{t}",
            )

            self.model.addConstr(self._b_c_311[t] <= aux1, name=f"bc311_ub1_{t}")
            self.model.addConstr(
                self._b_c_311[t] <= self.i_t310_active[t], name=f"bc311_ub2_{t}"
            )
            self.model.addConstr(
                self._b_c_311[t] >= aux1 + self.i_t310_active[t] - 1,
                name=f"bc311_lb_{t}",
            )

            c_tmp = self.model.addVar(
                vtype=GRB.INTEGER, lb=0, ub=self.n311_const, name=f"c311_tmp_{t}"
            )
            ut.lin_prod_bin_int(
                self.model,
                c_tmp,
                self._b_c_311[t],
                self.u_n311_sat[t],
                self.n311_const,
                name=f"c311_prod_{t}",
            )
            self.model.addConstr(self.c_n311[t] == c_tmp, name=f"c311_def_{t}")

        ub_u_n311 = self.n311_const + 1
        for t in range(self.n_steps):
            self.model.addConstr(
                self.u_n311_raw[t] >= self.n311_const * self.i_n311[t],
                name=f"i311_lb_{t}",
            )
            self.model.addConstr(
                self.u_n311_raw[t]
                <= (self.n311_const - 1)
                + (ub_u_n311 - (self.n311_const - 1)) * self.i_n311[t],
                name=f"i311_ub_{t}",
            )

        # T310 start
        self.model.addConstr(self.i_t310_start[0] <= self.i_n310[0], name="t310s0_ub1")
        self.model.addConstr(
            self.i_t310_start[0] <= 1 - self.i_rlf_running[0], name="t310s0_ub2"
        )
        self.model.addConstr(
            self.i_t310_start[0] >= self.i_n310[0] - self.i_rlf_running[0],
            name="t310s0_lb",
        )

        for t in range(1, self.n_steps):
            aux = self.model.addVar(vtype=GRB.BINARY, name=f"t310s_aux_{t}")
            self.model.addConstr(
                aux <= 1 - self.i_t310_active[t - 1], name=f"t310saux_ub1_{t}"
            )
            self.model.addConstr(
                aux <= 1 - self.i_rlf_running[t], name=f"t310saux_ub2_{t}"
            )
            self.model.addConstr(
                aux >= 1 - self.i_t310_active[t - 1] - self.i_rlf_running[t],
                name=f"t310saux_lb_{t}",
            )
            self.model.addConstr(
                self.i_t310_start[t] <= self.i_n310[t], name=f"t310s_ub1_{t}"
            )
            self.model.addConstr(self.i_t310_start[t] <= aux, name=f"t310s_ub2_{t}")
            self.model.addConstr(
                self.i_t310_start[t] >= self.i_n310[t] + aux - 1, name=f"t310s_lb_{t}"
            )

        # Stop conditions
        self.model.addConstr(self._i_t310_cancel_due_to_n311[0] == 0, name="cancel0")
        self.model.addConstr(self._i_t310_expiry[0] == 0, name="expiry0")
        self.model.addConstr(self.i_t310_stop[0] == 0, name="t310stop0")

        for t in range(1, self.n_steps):
            ut.lin_binary_and(
                self.model,
                self._i_t310_cancel_due_to_n311[t],
                self.i_t310_active[t - 1],
                self.i_n311[t],
                name=f"cancel_{t}",
            )
            ut.lin_binary_and(
                self.model,
                self._i_t310_expiry[t],
                self.i_t310_active[t - 1],
                self.i_rlf_running[t],
                name=f"expiry_{t}",
            )
            ut.lin_binary_or(
                self.model,
                self.i_t310_stop[t],
                [self._i_t310_cancel_due_to_n311[t], self._i_t310_expiry[t]],
                name=f"t310stop_{t}",
            )

        # Mode selection and tau update
        for t in range(self.n_steps):
            self.model.addConstr(
                self._m_rlf[t] == self.i_rlf_running[t], name=f"mrlf_{t}"
            )

            self.model.addConstr(
                self._m_start[t] <= self.i_t310_start[t], name=f"mstart_ub_{t}"
            )
            self.model.addConstr(
                self._m_start[t] <= 1 - self._m_rlf[t], name=f"mstart_norl_{t}"
            )
            self.model.addConstr(
                self._m_start[t] >= self.i_t310_start[t] - self._m_rlf[t],
                name=f"mstart_lb_{t}",
            )

            self.model.addConstr(
                self._m_stop[t] <= self.i_t310_stop[t], name=f"mstop_ub_{t}"
            )
            self.model.addConstr(
                self._m_stop[t] <= 1 - self._m_rlf[t], name=f"mstop_norl_{t}"
            )
            self.model.addConstr(
                self._m_stop[t] <= 1 - self._m_start[t], name=f"mstop_nostart_{t}"
            )
            self.model.addConstr(
                self._m_stop[t]
                >= self.i_t310_stop[t] - self._m_rlf[t] - self._m_start[t],
                name=f"mstop_lb_{t}",
            )

            self.model.addConstr(
                self._m_decr[t]
                == 1 - self._m_rlf[t] - self._m_start[t] - self._m_stop[t],
                name=f"mdecr_{t}",
            )

        # Initial tau
        self.model.addConstr(
            self.tau_t310[0] == self.t310_const * self.i_t310_start[0], name="tau0"
        )

        big_m_tau = self.t310_const
        for t in range(1, self.n_steps):
            m_zero = self.model.addVar(vtype=GRB.BINARY, name=f"mzero_{t}")
            ut.lin_binary_or(
                self.model,
                m_zero,
                [self._m_rlf[t], self._m_stop[t]],
                name=f"mzero_or_{t}",
            )

            self.model.addConstr(
                self.tau_t310[t] <= big_m_tau * (1 - m_zero), name=f"tau_zero_ub_{t}"
            )
            self.model.addConstr(self.tau_t310[t] >= 0, name=f"tau_zero_lb_{t}")

            self.model.addConstr(
                self.tau_t310[t]
                >= self.t310_const - big_m_tau * (1 - self._m_start[t]),
                name=f"tau_start_lb_{t}",
            )
            self.model.addConstr(
                self.tau_t310[t]
                <= self.t310_const + big_m_tau * (1 - self._m_start[t]),
                name=f"tau_start_ub_{t}",
            )

            expr = self.tau_t310[t - 1] - self.i_t310_active[t - 1]
            self.model.addConstr(
                self.tau_t310[t] - expr <= big_m_tau * (1 - self._m_decr[t]),
                name=f"tau_decr_ub_{t}",
            )
            self.model.addConstr(
                self.tau_t310[t] - expr >= -big_m_tau * (1 - self._m_decr[t]),
                name=f"tau_decr_lb_{t}",
            )

        # PCell change
        for n in range(self.n_bs):
            self.model.addConstr(
                self.i_same_cell[n, 0] == self.x_bs[n, 0], name=f"same0_{n}"
            )
            for t in range(1, self.n_steps):
                self.model.addConstr(
                    self.i_same_cell[n, t] <= self.x_bs[n, t - 1],
                    name=f"same_ub1_{n}_{t}",
                )
                self.model.addConstr(
                    self.i_same_cell[n, t] <= self.x_bs[n, t], name=f"same_ub2_{n}_{t}"
                )
                self.model.addConstr(
                    self.i_same_cell[n, t] >= self.x_bs[n, t - 1] + self.x_bs[n, t] - 1,
                    name=f"same_lb_{n}_{t}",
                )

        for t in range(self.n_steps):
            self.model.addConstr(
                self.i_pcell_change[t] == 1 - self.i_same_cell[:, t].sum(),
                name=f"pcell_change_{t}",
            )

        for i in range(self.n_steps - self.t_ho_const):
            self.model.addConstr(
                self.i_pcell_change[i : i + self.t_ho_const + 1].sum() <= 1,
                name=f"pcell_freq_{i}",
            )

        for t in range(1, self.n_steps):
            self.model.addConstr(
                self.i_pcell_change[t] + self.i_t310_active[t - 1] <= 1,
                name=f"pcell_no_t310_{t}",
            )

        for t in range(self.n_steps):
            self.model.addConstr(
                self.i_pcell_change[t] + self.i_rlf_running[t] <= 1,
                name=f"pcell_no_rlf_{t}",
            )
            self.model.addConstr(
                self.i_pcell_change[t] + self.i_q_out[t] <= 1,
                name=f"pcell_no_qout_{t}",
            )

        for t in range(1, self.n_steps):
            self.model.addConstr(
                self.i_pcell_change[t]
                <= self.i_rlf_running[t - 1] - self._i_rlf_in_last_t_ho_steps[t] + 1,
                name=f"pcell_after_rlf_{t}",
            )

        # HO execution
        for t in range(self.n_steps):
            self.model.addConstr(
                self.i_ho_exec_end[t] <= self.i_pcell_change[t], name=f"hoend_ub1_{t}"
            )
            self.model.addConstr(
                self.i_ho_exec_end[t] <= 1 - self._i_rlf_in_last_t_ho_steps[t],
                name=f"hoend_ub2_{t}",
            )
            self.model.addConstr(
                self.i_ho_exec_end[t]
                >= self.i_pcell_change[t] - self._i_rlf_in_last_t_ho_steps[t],
                name=f"hoend_lb_{t}",
            )

        for t in range(self.n_steps):
            tau_max = min(self.t_ho_exec_const, self.n_steps - t - 1)
            if tau_max == 0:
                self.model.addConstr(self.i_ho_exec_running[t] == 0, name=f"horun0_{t}")
                continue

            for tau in range(1, tau_max + 1):
                self.model.addConstr(
                    self.i_ho_exec_running[t] >= self.i_ho_exec_end[t + tau],
                    name=f"horun_lb_{t}_{tau}",
                )
            self.model.addConstr(
                self.i_ho_exec_running[t]
                <= self.i_ho_exec_end[t + 1 : t + tau_max + 1].sum(),
                name=f"horun_ub_{t}",
            )

        # Connectivity
        for t in range(self.n_steps):
            self.model.addConstr(
                self._b_conn_aux[t] <= 1 - self.i_ho_exec_running[t],
                name=f"bconn_ub1_{t}",
            )
            self.model.addConstr(
                self._b_conn_aux[t] <= 1 - self.i_rlf_running[t], name=f"bconn_ub2_{t}"
            )
            self.model.addConstr(
                self._b_conn_aux[t]
                >= 1 - self.i_ho_exec_running[t] - self.i_rlf_running[t],
                name=f"bconn_lb_{t}",
            )

        # Rate aggregation: s_rate[t] = sum_n cap[n,t] * x[n,t]
        capacity_mat = np.log2(1 + 10 ** (self.sinr_db / 10))
        cap_max_t = np.max(capacity_mat, axis=0)

        for t in range(self.n_steps):
            self.model.addConstr(
                self.s_rate[t] == (capacity_mat[:, t] * self.x_bs[:, t]).sum(),
                name=f"srate_def_{t}",
            )
            self.model.addConstr(
                self.r_rate[t] <= self.s_rate[t], name=f"rrate_ub_s_{t}"
            )
            self.model.addConstr(
                self.r_rate[t] <= cap_max_t[t] * self._b_conn_aux[t],
                name=f"rrate_ub_b_{t}",
            )
            self.model.addConstr(
                self.r_rate[t]
                >= self.s_rate[t] - cap_max_t[t] * (1 - self._b_conn_aux[t]),
                name=f"rrate_lb_{t}",
            )

        self._constraints_added = True

    def _set_objective(self) -> None:
        if not self._variables_added:
            raise RuntimeError("Variables must be added before setting objective.")
        if not self._constraints_added:
            raise RuntimeError("Constraints must be added before setting objective.")

        self.minimize_or_maximize = GRB.MAXIMIZE

        obj_rate = (1.0 / self.n_steps) * self.r_rate.sum()
        obj_out = (
            self.i_ho_exec_running.sum() + self.i_rlf_running.sum()
        ) / self.n_steps
        self.obj = obj_rate - self.lambda_r * obj_out
        self.model.setObjective(self.obj, GRB.MAXIMIZE)

    def extract_results(self) -> dict[str, np.ndarray]:
        if not self._optimization_finished:
            raise RuntimeError(
                "Optimization must be finished before extracting results."
            )

        pcells = np.argmax(self.x_bs.X, axis=0)

        results = {
            "sinr_db": np.asarray(self.sinr_db),
            "capacity": np.asarray(np.log2(1 + 10 ** (self.sinr_db / 10))),
            "pcell": np.asarray(pcells),
            "i_q_in": np.asarray(self.i_q_in.X),
            "i_q_out": np.asarray(self.i_q_out.X),
            "u_n310_raw": np.asarray(self.u_n310_raw.X),
            "c_n310": np.asarray(self.c_n310.X),
            "u_n311_raw": np.asarray(self.u_n311_raw.X),
            "c_n311": np.asarray(self.c_n311.X),
            "i_t310_start": np.asarray(self.i_t310_start.X),
            "i_t310_stop": np.asarray(self.i_t310_stop.X),
            "tau_t310_rem": np.asarray(self.tau_t310.X),
            "i_rlf_start": np.asarray(self.i_rlf_start.X),
            "i_ho_exec_end": np.asarray(self.i_ho_exec_end.X),
            "i_ho_exec_running": np.asarray(self.i_ho_exec_running.X),
            "i_t310_active": np.asarray(self.i_t310_active.X),
            "i_rlf_running": np.asarray(self.i_rlf_running.X),
        }
        return results

    def get_result_metrics(self) -> dict[str, float]:
        if not self._optimization_finished:
            raise RuntimeError(
                "Optimization must be finished before extracting metrics."
            )
        # Capacity metrics
        c_mean, max_c_mean = self._get_mean_capacity()

        # Connected time/outage metrics
        t_connected = self._get_connected_time()

        # Simulated time in seconds
        dt_s = self.n_steps * self.t_res_ms / 1_000

        # Number of (successful) handovers
        num_ho = int(np.sum(self.i_ho_exec_end.X))

        # Ping-pongs
        num_pp, pp_per_s = self._get_num_pp()

        metrics = {
            "simulation_id": self.rrc_config.simulation_id,
            "ue_idx": self.ue_idx,
            "simulated_time_s": dt_s,
            "simulated_steps": self.n_steps,
            "mean_capacity": c_mean,
            "max_mean_capacity": max_c_mean,
            "rel_connected_time": t_connected / self.n_steps,
            "num_ho": num_ho,
            "ho_per_s": num_ho / dt_s,
            "num_pp": num_pp,
            "pp_per_s": pp_per_s,
            "pp_rate": num_pp / num_ho if num_ho > 0 else float("inf"),
            "num_rlf": int(np.sum(self.i_rlf_start.X)),
            "ho_objective_value": self.obj.getValue(),
        }

        return metrics

    def _get_mean_capacity(self) -> tuple[float, float]:
        capacity_mat = np.log2(1 + 10 ** (self.sinr_db / 10))

        # Gurobi mean capacity
        pcells = np.argmax(self.x_bs.X, axis=0)
        c_sum = 0.0
        for t, pcell in enumerate(pcells):
            if self.i_ho_exec_running[t].X + self.i_rlf_running[t].X > 0.5:
                continue
            c_sum += capacity_mat[pcell, t]
        c_mean = c_sum / self.n_steps

        # Maximum mean capacity (no HO/RLF)
        max_c_mean = np.mean(np.max(capacity_mat, axis=0))

        return c_mean, max_c_mean

    def _get_connected_time(self) -> int:
        return self.n_steps - np.sum(
            self.i_ho_exec_running.X + self.i_rlf_running.X > 0.5
        )

    def _get_num_pp(self) -> tuple[int, float]:
        pcells = np.argmax(self.x_bs.X, axis=0)
        t_ho_end = np.where(self.i_ho_exec_end.X > 0.5)[0]

        num_pp = 0
        for t_ho1, t_ho2 in zip(t_ho_end[:-1], t_ho_end[1:]):
            if t_ho2 - t_ho1 <= self.t_mts_const:
                if pcells[t_ho1 - 1] == pcells[t_ho2]:
                    num_pp += 1

        dt_s = self.n_steps * self.t_res_ms / 1_000
        pp_per_s = num_pp / dt_s

        return num_pp, pp_per_s

    def log_results_to_csv(
        self, file_name: str | None = None, subfolder: str | None = None
    ) -> None:
        """Log optimization results to CSV file."""
        if not self._optimization_finished:
            raise RuntimeError("Optimization must be finished before logging results.")

        if file_name is None:
            file_name = (
                f"optim_{self.rrc_config.simulation_id}_"
                f"ep{self.data.ep_name}_ue{self.ue_idx}_lambda{self.lambda_r}.csv"
            )
            full_log_path = os.path.join(self.log_dir, file_name)

        if subfolder is not None:
            full_log_path = os.path.join(
                self.log_dir, subfolder, "full_results", file_name
            )
        else:
            full_log_path = os.path.join(self.log_dir, "full_results", file_name)
        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)
        file_exists = os.path.exists(full_log_path)

        with open(full_log_path, "a", newline="", encoding="utf-8") as f:
            header = self._get_snapshot(0).keys()
            writer = csv.DictWriter(f, fieldnames=header, delimiter=";")

            if not file_exists:
                writer.writeheader()

            for i in range(self.n_steps):
                row = self._get_snapshot(i)
                writer.writerow(row)

    def _get_snapshot(self, time_step: int) -> dict[str, Any]:
        if not self._optimization_finished:
            raise RuntimeError(
                "Optimization must be finished before extracting snapshot."
            )
        if time_step < 0 or time_step >= self.n_steps:
            raise ValueError("time_step is out of bounds.")

        snapshot = {
            "time_step": time_step,
            "ue_idx": self.ue_idx,
            "pcell_gurobi": int(np.argmax(np.asarray(self.x_bs.X)[:, time_step])),
            "i_q_in": int(np.asarray(self.i_q_in.X)[time_step]),
            "i_q_out": int(np.asarray(self.i_q_out.X)[time_step]),
            "u_n310_raw": int(np.asarray(self.u_n310_raw.X)[time_step]),
            "c_n310": int(np.asarray(self.c_n310.X)[time_step]),
            "u_n311_raw": int(np.asarray(self.u_n311_raw.X)[time_step]),
            "c_n311": int(np.asarray(self.c_n311.X)[time_step]),
            "i_t310_start": int(np.asarray(self.i_t310_start.X)[time_step]),
            "i_t310_stop": int(np.asarray(self.i_t310_stop.X)[time_step]),
            "tau_t310_rem": int(np.asarray(self.tau_t310.X)[time_step]),
            "_i_t310_eq_1": int(np.asarray(self._i_t310_eq_1.X)[time_step]),
            "i_rlf_start": int(np.asarray(self.i_rlf_start.X)[time_step]),
            "i_ho_exec_end": int(np.asarray(self.i_ho_exec_end.X)[time_step]),
            "i_ho_exec_running": int(np.asarray(self.i_ho_exec_running.X)[time_step]),
            "i_t310_active": int(np.asarray(self.i_t310_active.X)[time_step]),
            "i_rlf_running": int(np.asarray(self.i_rlf_running.X)[time_step]),
            "sinr_db": np.asarray(self.sinr_db)[:, time_step].tolist(),
            "capacity": np.log2(
                1 + 10 ** (np.asarray(self.sinr_db)[:, time_step] / 10)
            ).tolist(),
        }
        return snapshot
