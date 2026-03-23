"""MILP optimizer implementation using Gurobi."""

import multiprocessing
import os
from typing import Any, TYPE_CHECKING
from warnings import warn

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .base_optimizer import GurobiBaseOptimizer
from . import milp_variables as milp_vars
from .milp_constraints import MILPConstraintBuilder
from .milp_dataclasses import (
    ConstraintContext,
    ProblemData,
    RRCConstants,
    WarmStartConfig,
)
from .milp_result_extractor import ResultExtractor
from ..dataloader.preprocessing import preprocess_dataset
from ..reference.reference import RRCReferenceSimulation

if TYPE_CHECKING:
    from .optim_config import OptimConfig
    from ..result_manager.episode_result import EpisodeResult


class HandoverOptimizerMILP(GurobiBaseOptimizer):
    """Gurobi-based MILP handover optimizer."""

    ep_result: "EpisodeResult"
    data: ProblemData
    warm_start_config: WarmStartConfig
    vars: milp_vars.ModelVars
    _extractor: ResultExtractor | None

    def __init__(
        self,
        config: "OptimConfig",
        warm_start_config: WarmStartConfig | None = None,
    ) -> None:
        """Initialise the optimizer."""
        super().__init__(config)

        # RRC Parameters
        t_res_ms = config.rrc.t_res_ms
        self.rrc_consts = RRCConstants(
            t_res_ms=t_res_ms,
            q_in_db=config.rrc.q_in_db,
            q_out_db=config.rrc.q_out_db,
            n310=config.rrc.n310,
            n311=config.rrc.n311,
            t310=config.rrc.t310_ms // t_res_ms,
            t_ho_prep_sim=config.rrc.t_ho_prep_ms_simulated // t_res_ms,
            t_ho_exec_sim=config.rrc.t_ho_exec_ms_simulated // t_res_ms,
            t_rlfr_sim=config.rrc.t_rlfr_ms_simulated // t_res_ms,
            t_mts=config.rrc.t_mts_ms // t_res_ms,
        )

        # Warm start/safe candidate pruning configuration
        self.warm_start_config = warm_start_config or WarmStartConfig()
        self._extractor = None

    def load_data(self, ep_result: "EpisodeResult", ue_idx: int) -> None:
        """Data loading."""
        self.ep_result = ep_result
        rsrp_dbm, sinr_db, q_in_mat, q_out_mat = preprocess_dataset(
            ep_result, self.rrc_config, use_l3_filtering=True, ue_no=ue_idx
        )

        n_bs, n_steps = sinr_db.shape
        self.data = ProblemData(
            ue_idx=ue_idx,
            n_bs=n_bs,
            n_steps=n_steps,
            rsrp_dbm=rsrp_dbm,
            sinr_db=sinr_db,
            q_in_mat=q_in_mat,
            q_out_mat=q_out_mat,
        )

        self._data_loaded = True

    def setup_model(self, name: str = "HandoverOptimizerMILP") -> None:
        """Build the MILP model with variables, constraints, and objective."""
        if not self._data_loaded:
            raise RuntimeError("Data must be loaded before building the model.")

        # Build final, unpruned model -> optimality preserved regardless of warm start
        self.init_model(name=name)
        self.model.setParam("Threads", multiprocessing.cpu_count())

        self._add_variables()
        self._add_constraints()
        self._set_objective()

    def apply_warm_start_if_configured(self) -> None:
        """Apply warm start if it was computed during initialization."""
        if self.ep_result is None:
            raise RuntimeError("Data must be loaded before applying warm start.")
        if not self._variables_added:
            raise RuntimeError("Variables must be added before applying warm start.")

        _warm_start_x = self._solve_pruned_for_warm_start()
        if _warm_start_x is None:
            warn(
                "No warm start solution found or warm start not configured. "
                "Proceeding without warm start."
            )
        else:
            self._apply_warm_start(_warm_start_x)

    def solve(self) -> None:
        """Run the solver and build the ResultExtractor for post-processing."""
        super().solve()
        if self._optimization_finished:
            log_dir = os.path.join("results", "optimization")
            os.makedirs(log_dir, exist_ok=True)
            self._extractor = ResultExtractor(
                self.vars,
                self.data,
                self.rrc_consts,
                simulation_id=self.rrc_config.simulation_id,
                lambda_r=self.config.lambda_r,
                ep_name=self.ep_result.ep_name,
                log_dir=log_dir,
            )

    def extract_results(self) -> dict[str, np.ndarray]:
        """Extract optimization results as numpy arrays."""
        if not self._optimization_finished or self._extractor is None:
            raise RuntimeError("Optimization must finish before extracting results.")
        return self._extractor.extract_results()

    def get_result_metrics(self) -> dict[int, dict[str, Any]]:
        """Get key optimization metrics in a structured format."""
        if not self._optimization_finished or self._extractor is None:
            raise RuntimeError("Optimization must finish before extracting metrics.")
        return self._extractor.get_result_metrics(obj_value=float(self.obj.getValue()))

    def get_aggregated_result_metrics(self) -> dict[str, Any]:
        """
        Get key optimization metrics in a structured format.

        Basically the same as `get_result_metrics` since the MILP only optimizes
        for a single UE, but provided for consistency with the reference
        simulation metrics.
        """
        if not self._optimization_finished or self._extractor is None:
            raise RuntimeError("Optimization must finish before extracting metrics.")
        return self.get_result_metrics()[self.data.ue_idx]

    def log_results_to_csv(
        self, file_name: str | None = None, subfolder: str | None = None
    ) -> None:
        """Log optimization results to CSV file."""
        if not self._optimization_finished or self._extractor is None:
            raise RuntimeError("Optimization must finish before logging results.")
        self._extractor.log_results_to_csv(file_name=file_name, subfolder=subfolder)

    def _add_constraints(self) -> None:
        """Assemble and add all MILP constraints to the model."""
        if not self._variables_added:
            raise RuntimeError("Variables must be added before adding constraints.")

        ctx = ConstraintContext(
            model=self.model,
            vars=self.vars,
            data=self.data,
            consts=self.rrc_consts,
            debug=self.config.debug,
            ref_results=self._get_reference_result(),
        )

        cs_builder = MILPConstraintBuilder(ctx)

        cs_builder.add_constraints()
        self._constraints_added = cs_builder.constraints_added

    def _add_variables(self) -> None:
        """Allocate all Gurobi decision variables and attach them to the model."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before adding variables.")
        self.vars = self._build_model_vars()
        self.model.update()
        self._variables_added = True

    def _apply_warm_start(self, x_start: np.ndarray) -> None:
        """Set MIP start hints on ``x_bs`` from a precomputed one-hot assignment.

        Parameters
        ----------
        x_start:
            ``(n_bs, n_steps)`` binary array with exactly one 1 per column.
        """
        if x_start.shape != self.vars.x_bs.shape:
            raise ValueError(
                f"x_start shape {x_start.shape} does not match ({self.vars.x_bs.shape})"
            )

        for n in range(self.data.n_bs):
            for t in range(self.data.n_steps):
                self.vars.x_bs[n, t].Start = float(x_start[n, t])

    def _build_model_vars(self) -> milp_vars.ModelVars:
        """Allocate all Gurobi ``MVar`` objects and return them as a frozen ``ModelVars`` tree."""
        model = self.model
        n_bs = self.data.n_bs
        n_steps = self.data.n_steps
        const = self.rrc_consts

        return milp_vars.ModelVars(
            x_bs=model.addMVar((n_bs, n_steps), vtype=GRB.BINARY, name="x_bs"),
            i_q_in=model.addMVar(n_steps, vtype=GRB.BINARY, name="i_q_in"),
            i_q_out=model.addMVar(n_steps, vtype=GRB.BINARY, name="i_q_out"),
            n310=milp_vars.CounterVars(
                u_raw=model.addMVar(
                    n_steps,
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=const.n310 + 1,
                    name="n310_u_raw",
                ),
                u_sat=model.addMVar(
                    n_steps, vtype=GRB.INTEGER, lb=0, ub=const.n310, name="n310_u_sat"
                ),
                cnt=model.addMVar(
                    n_steps, vtype=GRB.INTEGER, lb=0, ub=const.n310, name="n310_c"
                ),
                ind=model.addMVar(n_steps, vtype=GRB.BINARY, name="n310_i"),
                b_reset=model.addMVar(n_steps, vtype=GRB.BINARY, name="n310_b_reset"),
                b_q=model.addMVar(n_steps, vtype=GRB.BINARY, name="n310_b_q"),
                b_c=model.addMVar(n_steps, vtype=GRB.BINARY, name="n310_b_c"),
            ),
            n311=milp_vars.CounterVars(
                u_raw=model.addMVar(
                    n_steps,
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=const.n311 + 1,
                    name="n311_u_raw",
                ),
                u_sat=model.addMVar(
                    n_steps, vtype=GRB.INTEGER, lb=0, ub=const.n311, name="n311_u_sat"
                ),
                cnt=model.addMVar(
                    n_steps, vtype=GRB.INTEGER, lb=0, ub=const.n311, name="n311_c"
                ),
                ind=model.addMVar(n_steps, vtype=GRB.BINARY, name="n311_i"),
                b_reset=model.addMVar(n_steps, vtype=GRB.BINARY, name="n311_b_reset"),
                b_q=model.addMVar(n_steps, vtype=GRB.BINARY, name="n311_b_q"),
                b_c=model.addMVar(n_steps, vtype=GRB.BINARY, name="n311_b_c"),
            ),
            t310=milp_vars.T310Vars(
                start=model.addMVar(n_steps, vtype=GRB.BINARY, name="t310_start"),
                stop=model.addMVar(n_steps, vtype=GRB.BINARY, name="t310_stop"),
                active=model.addMVar(n_steps, vtype=GRB.BINARY, name="t310_active"),
                cancel_due_to_n311=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="t310_cancel_due_to_n311"
                ),
                expiry=model.addMVar(n_steps, vtype=GRB.BINARY, name="t310_expiry"),
                tau=model.addMVar(
                    n_steps, vtype=GRB.INTEGER, lb=0, ub=const.t310, name="t310_tau"
                ),
                tau_eq_1=model.addMVar(n_steps, vtype=GRB.BINARY, name="t310_tau_eq_1"),
                b_mode_rlf=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="t310_b_mode_rlf"
                ),
                b_mode_start=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="t310_b_mode_start"
                ),
                b_mode_stop=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="t310_b_mode_stop"
                ),
                b_mode_decr=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="t310_b_mode_decr"
                ),
            ),
            rlf=milp_vars.RLFVars(
                start=model.addMVar(n_steps, vtype=GRB.BINARY, name="rlf_start"),
                running=model.addMVar(n_steps, vtype=GRB.BINARY, name="rlf_running"),
                in_last_t_ho_steps=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="rlf_in_last_t_ho_steps"
                ),
            ),
            ho=milp_vars.HandoverVars(
                same_cell=model.addMVar(
                    (n_bs, n_steps), vtype=GRB.BINARY, name="b_same_cell"
                ),
                cell_change=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="b_cell_change"
                ),
                exec_end=model.addMVar(n_steps, vtype=GRB.BINARY, name="ho_exec_end"),
                exec_running=model.addMVar(
                    n_steps, vtype=GRB.BINARY, name="ho_exec_running"
                ),
            ),
            rate=milp_vars.RateVars(
                conn_aux=model.addMVar(n_steps, vtype=GRB.BINARY, name="b_conn_aux"),
                s_rate=model.addMVar(
                    n_steps, vtype=GRB.CONTINUOUS, lb=0.0, name="s_rate"
                ),
                r_rate=model.addMVar(
                    n_steps, vtype=GRB.CONTINUOUS, lb=0.0, name="r_rate"
                ),
            ),
        )

    def _compute_candidate_sets(self) -> list[np.ndarray]:
        """Candidate pruning (safe warm start)"""
        metric = self.warm_start_config.candidate_metric.lower()
        if metric == "sinr":
            score = self.data.sinr_db
        else:
            score = self.data.rsrp_dbm

        candidates: list[np.ndarray] = []
        prev_cand: set[int] = set()

        for t in range(self.data.n_steps):
            s = score[:, t]
            idx = np.arange(self.data.n_bs)

            cand_set: set[int] = set()
            if self.warm_start_config.candidate_k > 0:
                k = min(self.warm_start_config.candidate_k, self.data.n_bs)
                topk = idx[np.argpartition(-s, k - 1)[:k]]
                cand_set.update(int(i) for i in topk)

            if self.warm_start_config.candidate_delta_db > 0.0:
                best = float(np.max(s))
                mask = s >= (best - self.warm_start_config.candidate_delta_db)
                cand_set.update(int(i) for i in idx[mask])

            if not cand_set:
                cand_set.add(int(np.argmax(s)))

            if self.warm_start_config.candidate_include_prev and prev_cand:
                cand_set.update(prev_cand)

            candidates.append(np.array(sorted(cand_set), dtype=int))
            prev_cand = cand_set

        return candidates

    def _get_reference_result(self) -> dict[str, np.ndarray] | None:
        """Run the reference RRC simulation and return its results when in debug mode.

        Returns ``None`` in normal (non-debug) operation so that no reference
        labels are injected into the constraint set.
        """
        if self.config.debug:
            sim = RRCReferenceSimulation(self.config.rrc, self.ep_result)
            sim.run()
            return sim.extract_results()
        return None

    def _set_objective(self) -> None:
        """Define and register the optimization objective."""
        if not self._variables_added:
            raise RuntimeError("Variables must be added before setting objective.")
        if not self._constraints_added:
            raise RuntimeError("Constraints must be added before setting objective.")

        obj_rate = self.vars.rate.r_rate.sum() / self.data.n_steps
        obj_out = (
            self.vars.ho.exec_running.sum() + self.vars.rlf.running.sum()
        ) / self.data.n_steps
        self.obj = obj_rate - self.config.lambda_r * obj_out

        if self.config.debug:
            self.model.setObjective(0.0, GRB.MAXIMIZE)
        else:
            self.model.setObjective(self.obj, GRB.MAXIMIZE)

        self._objective_set = True

    def _solve_pruned_for_warm_start(self) -> np.ndarray | None:
        """Solve a candidate-pruned sub-model to obtain a warm-start assignment."""
        if (
            self.warm_start_config.candidate_k <= 0
            and self.warm_start_config.candidate_delta_db <= 0.0
        ):
            return None  # No pruning criteria specified, skip warm start

        candidates = self._compute_candidate_sets()

        pruned = gp.Model("HandoverOptimizerMILP_pruned")
        pruned.setParam("Threads", multiprocessing.cpu_count())
        pruned.setParam("OutputFlag", 0)

        if (
            self.warm_start_config.safe_pruning_time_limit_s is not None
            and self.warm_start_config.safe_pruning_time_limit_s > 0
        ):
            pruned.setParam(
                "TimeLimit", self.warm_start_config.safe_pruning_time_limit_s
            )
        if (
            self.warm_start_config.safe_pruning_mip_gap is not None
            and self.warm_start_config.safe_pruning_mip_gap > 0
        ):
            pruned.setParam("MIPGap", self.warm_start_config.safe_pruning_mip_gap)

        ub_x = np.zeros((self.data.n_bs, self.data.n_steps), dtype=float)
        for t, cand in enumerate(candidates):
            ub_x[cand, t] = 1.0

        x_bs = pruned.addMVar(
            (self.data.n_bs, self.data.n_steps), vtype=GRB.BINARY, ub=ub_x, name="x_bs"
        )

        # Quick, logic-free warm start objective: maximize average capacity under
        # connectivity proxy. This does not alter final optimality because it is
        # only for warm start.
        pruned.addConstr(x_bs.sum(axis=0) == 1, name="onehot")
        obj = (self.data.capacity * x_bs).sum() / self.data.n_steps
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
        warm[pcells, np.arange(self.data.n_steps)] = 1.0
        return warm
