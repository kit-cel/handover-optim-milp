"""Constraints for the handover optimization MILP model.

The main blocks correspond to the published formulation:

* Eq. (9): selected-cell Q-in/Q-out indicators.
* Eqs. (10a)-(10c): RLF start and recovery state.
* Eqs. (11a)-(11e): N310 out-of-sync counter logic.
* Eqs. (12a)-(12e): N311 in-sync counter logic.
* Eqs. (13a)-(13f): T310 start/cancel/expiry/stop and timer update.
* Eqs. (14a)-(14d): PCell-change detection and admissibility.
* Eqs. (15a)-(15b): HO completion and execution-running indicators.

Several products, indicators, maxima, and minima from the paper are expanded
into auxiliary binary/integer variables and linear inequalities.
"""

import numpy as np
from gurobipy import GRB

from . import optim_utils as ut
from .milp_dataclasses import ConstraintContext


class MILPConstraintBuilder:
    """Builds all constraints for the handover optimization MILP model."""

    def __init__(self, ctx: ConstraintContext) -> None:
        """Store the constraint context and initialise bookkeeping state."""
        self.ctx = ctx

        self.n_bs = ctx.data.n_bs
        self.n_steps = ctx.data.n_steps

        self._constraints_added = False

    def add_constraints(self) -> None:
        """Add all constraints to the model."""
        self._add_pcell_constraints()
        self._add_n310_constraints()
        self._add_n311_constraints()
        self._add_t310_constraints()
        self._add_rlf_constraints()
        self._add_handover_constraints()
        self._add_rate_constraints()

        self._constraints_added = True

    def _add_pcell_constraints(self) -> None:
        """Add primary-cell (PCell) constraints.

        Enforces one selected serving cell per time step, as described in
        Sec. III-C, and implements Eq. (9) for the Q-in/Q-out indicators of
        the selected serving cell.
        """
        d = self.ctx.data
        m = self.ctx.model
        v = self.ctx.vars

        # one-hot PCell
        m.addConstr(v.x_bs.sum(axis=0) == 1, name="pcell_onehot")
        if self.ctx.initial_access == "strongest_rsrp":
            strongest = int(np.argmax(d.rsrp_dbm[:, 0]))
            m.addConstr(
                v.x_bs[strongest, 0] == 1,
                name="pcell_causal_initial_access",
            )
        elif self.ctx.initial_access != "free":
            raise ValueError(f"unknown initial access mode: {self.ctx.initial_access}")

        # Q_in/Q_out indicators. The per-step dot product x_t^T q_t is the
        # column-wise sum of the elementwise product, so the whole of
        # Eq. (9) is two array constraints instead of 2*T scalar ones.
        m.addConstr(v.i_q_in == (d.q_in_mat * v.x_bs).sum(axis=0), name="q_in_def")
        m.addConstr(v.i_q_out == (d.q_out_mat * v.x_bs).sum(axis=0), name="q_out_def")
        # # Initial condition that forces T310 and related counters and timers to be inactive at
        # # t=0, which is a reasonable assumption for the start of the simulation.
        # # It would prevent selecting an initially out-of-sync PCell and thereby avoid
        # # the current t=0 N310/T310 start path, but it is disabled to preserve feasibility.
        # m.addConstr(v.i_q_out[0] == 0, name="qout_initial_zero")

    def _add_n310_constraints(self) -> None:
        """Add N310 out-of-sync counter constraints.

        Implements Eqs. (11a)-(11e). The N310 counter increments on selected
        serving-cell Q-out indications while T310 is not already active, is
        disabled during HO/RLF interruption, is saturated at ``N310``, and
        exposes the threshold indicator used to start T310.
        """
        n_steps = self.n_steps
        m = self.ctx.model
        v = self.ctx.vars
        c = self.ctx.consts

        # Constraints that create no variables are written as whole-array
        # operations; the per-step loops below are kept where they add
        # auxiliary variables, so the model's column order -- and with it
        # which degenerate optimum the solver settles on -- is unchanged.
        # Eq. (11a): b_reset = (1 - i_RLF,run) AND (1 - i_HO,run)
        m.addConstr(v.n310.b_reset <= 1 - v.rlf.running, name="reset310_ub1")
        m.addConstr(v.n310.b_reset <= 1 - v.ho.exec_running, name="reset310_ub2")
        m.addConstr(
            v.n310.b_reset >= 1 - v.rlf.running - v.ho.exec_running,
            name="reset310_lb",
        )
        # Eq. (11d)'s gate: b_c = b_reset AND (1 - i_T310,run)
        m.addConstr(v.n310.b_c <= v.n310.b_reset, name="c310_ub1")
        m.addConstr(v.n310.b_c <= 1 - v.t310.active, name="c310_ub2")
        m.addConstr(v.n310.b_c >= v.n310.b_reset - v.t310.active, name="c310_lb")

        # Eq. (11b)'s increment: b_q = i_out AND (1 - i_T310,run_{t-1})
        m.addConstr(v.n310.b_q[0] == v.i_q_out[0], name="bq310_0")
        if n_steps > 1:
            m.addConstr(v.n310.b_q[1:] <= v.i_q_out[1:], name="bq310_ub1")
            m.addConstr(v.n310.b_q[1:] <= 1 - v.t310.active[:-1], name="bq310_ub2")
            m.addConstr(
                v.n310.b_q[1:] >= v.i_q_out[1:] - v.t310.active[:-1],
                name="bq310_lb",
            )

        m.addConstr(v.n310.u_raw[0] == v.i_q_out[0], name="u310_0")

        for t in range(1, n_steps):
            c_tmp = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=c.n310 + 1,
                name=f"temp310_{t}",
            )
            m.addConstr(
                c_tmp == v.n310.cnt[t - 1] + v.n310.b_q[t],
                name=f"temp310_def_{t}",
            )

            u_raw_aux = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=c.n310 + 1,
                name=f"u310_rawtmp_{t}",
            )
            ut.lin_prod_bin_int(
                m,
                u_raw_aux,
                v.n310.b_reset[t],
                c_tmp,
                c.n310 + 1,
                name=f"u310_prod_{t}",
            )
            m.addConstr(v.n310.u_raw[t] == u_raw_aux, name=f"u310_raw_def_{t}")

        for t in range(n_steps):
            ut.lin_saturate_min(
                m,
                v.n310.u_sat[t],
                v.n310.u_raw[t],
                c.n310,
                c.n310 + 1,
                name=f"u310_sat_{t}",
            )

        for t in range(n_steps):
            c_tmp = m.addVar(vtype=GRB.INTEGER, lb=0, ub=c.n310, name=f"c310_tmp_{t}")
            ut.lin_prod_bin_int(
                m,
                c_tmp,
                v.n310.b_c[t],
                v.n310.u_sat[t],
                c.n310,
                name=f"c310_prod_{t}",
            )
            m.addConstr(v.n310.cnt[t] == c_tmp, name=f"c310_def_{t}")

        # Eq. (11e): i^310 = 1{u^310 >= N310}
        ub_u_n310 = c.n310 + 1
        m.addConstr(v.n310.u_raw >= c.n310 * v.n310.ind, name="i310_lb")
        m.addConstr(
            v.n310.u_raw <= (c.n310 - 1) + (ub_u_n310 - (c.n310 - 1)) * v.n310.ind,
            name="i310_ub",
        )

    def _add_n311_constraints(self) -> None:
        """Add N311 in-sync counter constraints.

        Implements Eqs. (12a)-(12e). The N311 counter increments on selected
        serving-cell Q-in indications while T310 was active in the previous
        step, is disabled during HO/RLF interruption, and exposes the
        threshold indicator used to cancel T310.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # Constraints that create no variables are written as whole-array
        # operations; the per-step loops below are kept where they add
        # auxiliary variables, so the model's column order -- and with it
        # which degenerate optimum the solver settles on -- is unchanged.
        # Eq. (12a)
        m.addConstr(v.n311.b_reset <= 1 - v.rlf.running, name="reset311_ub1")
        m.addConstr(v.n311.b_reset <= 1 - v.ho.exec_running, name="reset311_ub2")
        m.addConstr(
            v.n311.b_reset >= 1 - v.rlf.running - v.ho.exec_running,
            name="reset311_lb",
        )

        # Eq. (12b)'s increment: b_q = i_in AND i_T310,run_{t-1}. Note the
        # asymmetry with N310: this one counts only *while* T310 runs.
        m.addConstr(v.n311.b_q[0] == 0, name="bq311_0")
        if n_steps > 1:
            m.addConstr(v.n311.b_q[1:] <= v.i_q_in[1:], name="bq311_ub1")
            m.addConstr(v.n311.b_q[1:] <= v.t310.active[:-1], name="bq311_ub2")
            m.addConstr(
                v.n311.b_q[1:] >= v.i_q_in[1:] + v.t310.active[:-1] - 1,
                name="bq311_lb",
            )

        m.addConstr(v.n311.u_raw[0] == 0, name="u311_0")
        m.addConstr(v.n311.cnt[0] == 0, name="c311_0")

        for t in range(1, n_steps):
            c_tmp = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=c.n311 + 1,
                name=f"temp311_{t}",
            )
            m.addConstr(
                c_tmp == v.n311.cnt[t - 1] + v.n311.b_q[t],
                name=f"temp311_def_{t}",
            )

            u_raw_aux = m.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=c.n311 + 1,
                name=f"u311_rawtmp_{t}",
            )
            ut.lin_prod_bin_int(
                m,
                u_raw_aux,
                v.n311.b_reset[t],
                c_tmp,
                c.n311 + 1,
                name=f"u311_prod_{t}",
            )
            m.addConstr(v.n311.u_raw[t] == u_raw_aux, name=f"u311_raw_def_{t}")

        for t in range(n_steps):
            ut.lin_saturate_min(
                m,
                v.n311.u_sat[t],
                v.n311.u_raw[t],
                c.n311,
                c.n311 + 1,
                name=f"u311_sat_{t}",
            )

        # b_c_311[t] = b_reset_311[t] AND (1 - i_t310_stop[t]) AND i_t310_active[t]
        for t in range(n_steps):
            aux1 = m.addVar(vtype=GRB.BINARY, name=f"aux311_{t}")
            m.addConstr(aux1 <= v.n311.b_reset[t], name=f"aux311_ub1_{t}")
            m.addConstr(aux1 <= 1 - v.t310.stop[t], name=f"aux311_ub2_{t}")
            m.addConstr(
                aux1 >= v.n311.b_reset[t] - v.t310.stop[t],
                name=f"aux311_lb_{t}",
            )

            m.addConstr(v.n311.b_c[t] <= aux1, name=f"bc311_ub1_{t}")
            m.addConstr(v.n311.b_c[t] <= v.t310.active[t], name=f"bc311_ub2_{t}")
            m.addConstr(
                v.n311.b_c[t] >= aux1 + v.t310.active[t] - 1,
                name=f"bc311_lb_{t}",
            )

            c_tmp = m.addVar(vtype=GRB.INTEGER, lb=0, ub=c.n311, name=f"c311_tmp_{t}")
            ut.lin_prod_bin_int(
                m,
                c_tmp,
                v.n311.b_c[t],
                v.n311.u_sat[t],
                c.n311,
                name=f"c311_prod_{t}",
            )
            m.addConstr(v.n311.cnt[t] == c_tmp, name=f"c311_def_{t}")

        # Eq. (12e)
        ub_u_n311 = c.n311 + 1
        m.addConstr(v.n311.u_raw >= c.n311 * v.n311.ind, name="i311_lb")
        m.addConstr(
            v.n311.u_raw <= (c.n311 - 1) + (ub_u_n311 - (c.n311 - 1)) * v.n311.ind,
            name="i311_ub",
        )

    def _add_t310_constraints(self) -> None:
        """Add T310 timer constraints.

        Implements Eqs. (13a)-(13f). The paper's products and max operator
        are encoded with auxiliary binary mode variables. ``tau_eq_1`` is an
        implementation helper for the expiry condition ``tau[t] == 1``.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # Eq. (13f): i^{T310,run} = 1{tau >= 1}, encoded exactly via the
        # integer countdown. Whole-array form; no variables are created
        # here, so the model's column order is untouched.
        m.addConstr(v.t310.tau >= v.t310.active, name="t310_act_lb")
        m.addConstr(v.t310.tau <= c.t310 * v.t310.active, name="t310_act_ub")

        # tau_eq_1 = 1 <=> tau == 1, the "about to expire" helper Eq. (10a)
        # reads one step later.
        z = v.t310.tau_eq_1
        m.addConstr(v.t310.tau - 1 <= (c.t310 - 1) * (1 - z), name="t310_eq1_ub")
        m.addConstr(1 - v.t310.tau <= 1 * (1 - z), name="t310_eq1_lb")
        if c.t310 >= 2:
            # tau = 1 => z = 1; 'active' eliminates the tau = 0 case.
            m.addConstr(v.t310.tau >= 2 * (v.t310.active - z), name="t310_eq1_rev")

        # T310 start
        m.addConstr(v.t310.start[0] <= v.n310.ind[0], name="t310s0_ub1")
        m.addConstr(v.t310.start[0] <= 1 - v.rlf.running[0], name="t310s0_ub2")
        m.addConstr(
            v.t310.start[0] >= v.n310.ind[0] - v.rlf.running[0],
            name="t310s0_lb",
        )

        for t in range(1, n_steps):
            aux = m.addVar(vtype=GRB.BINARY, name=f"t310s_aux_{t}")
            m.addConstr(aux <= 1 - v.t310.active[t - 1], name=f"t310saux_ub1_{t}")
            m.addConstr(aux <= 1 - v.rlf.running[t], name=f"t310saux_ub2_{t}")
            m.addConstr(
                aux >= 1 - v.t310.active[t - 1] - v.rlf.running[t],
                name=f"t310saux_lb_{t}",
            )
            m.addConstr(
                v.t310.start[t] <= v.n310.ind[t],
                name=f"t310s_ub1_{t}",
            )
            m.addConstr(v.t310.start[t] <= aux, name=f"t310s_ub2_{t}")
            m.addConstr(
                v.t310.start[t] >= v.n310.ind[t] + aux - 1,
                name=f"t310s_lb_{t}",
            )

        # Stop conditions
        m.addConstr(v.t310.cancel_due_to_n311[0] == 0, name="cancel0")
        m.addConstr(v.t310.expiry[0] == 0, name="expiry0")
        m.addConstr(v.t310.stop[0] == 0, name="t310stop0")

        for t in range(1, n_steps):
            ut.lin_binary_and(
                m,
                v.t310.cancel_due_to_n311[t],
                v.t310.active[t - 1],
                v.n311.ind[t],
                name=f"cancel_{t}",
            )
            ut.lin_binary_and(
                m,
                v.t310.expiry[t],
                v.t310.active[t - 1],
                v.rlf.running[t],
                name=f"expiry_{t}",
            )
            ut.lin_binary_or(
                m,
                v.t310.stop[t],
                [v.t310.cancel_due_to_n311[t], v.t310.expiry[t]],
                name=f"t310stop_{t}",
            )

        # Mode selection for the Eq. (13e) timer update: the paper's
        # nested expression is expanded into four mutually exclusive
        # modes. Their equivalence rests on start and stop never
        # coinciding, which holds structurally and is pinned by
        # tests/ho_optim_milp/test_milp_paper_equations.py.
        m.addConstr(v.t310.b_mode_rlf == v.rlf.running, name="mrlf")

        m.addConstr(v.t310.b_mode_start <= v.t310.start, name="mstart_ub")
        m.addConstr(v.t310.b_mode_start <= 1 - v.t310.b_mode_rlf, name="mstart_norl")
        m.addConstr(
            v.t310.b_mode_start >= v.t310.start - v.t310.b_mode_rlf,
            name="mstart_lb",
        )

        m.addConstr(v.t310.b_mode_stop <= v.t310.stop, name="mstop_ub")
        m.addConstr(v.t310.b_mode_stop <= 1 - v.t310.b_mode_rlf, name="mstop_norl")
        m.addConstr(v.t310.b_mode_stop <= 1 - v.t310.b_mode_start, name="mstop_nostart")
        m.addConstr(
            v.t310.b_mode_stop >= v.t310.stop - v.t310.b_mode_rlf - v.t310.b_mode_start,
            name="mstop_lb",
        )

        m.addConstr(
            v.t310.b_mode_decr
            == 1 - v.t310.b_mode_rlf - v.t310.b_mode_start - v.t310.b_mode_stop,
            name="mdecr",
        )

        # Initial tau
        m.addConstr(
            v.t310.tau[0] == c.t310 * v.t310.start[0],
            name="tau0",
        )

        big_m_tau = c.t310
        for t in range(1, n_steps):
            m_zero = m.addVar(vtype=GRB.BINARY, name=f"mzero_{t}")
            ut.lin_binary_or(
                m,
                m_zero,
                [v.t310.b_mode_rlf[t], v.t310.b_mode_stop[t]],
                name=f"mzero_or_{t}",
            )

            m.addConstr(
                v.t310.tau[t] <= big_m_tau * (1 - m_zero),
                name=f"tau_zero_ub_{t}",
            )
            m.addConstr(v.t310.tau[t] >= 0, name=f"tau_zero_lb_{t}")

            m.addConstr(
                v.t310.tau[t] >= c.t310 - big_m_tau * (1 - v.t310.b_mode_start[t]),
                name=f"tau_start_lb_{t}",
            )
            m.addConstr(
                v.t310.tau[t] <= c.t310 + big_m_tau * (1 - v.t310.b_mode_start[t]),
                name=f"tau_start_ub_{t}",
            )

            expr = v.t310.tau[t - 1] - v.t310.active[t - 1]
            m.addConstr(
                v.t310.tau[t] - expr <= big_m_tau * (1 - v.t310.b_mode_decr[t]),
                name=f"tau_decr_ub_{t}",
            )
            m.addConstr(
                v.t310.tau[t] - expr >= -big_m_tau * (1 - v.t310.b_mode_decr[t]),
                name=f"tau_decr_lb_{t}",
            )

    def _add_rlf_constraints(self) -> None:
        """Add Radio Link Failure (RLF) recovery state constraints.

        Implements Eqs. (10a)-(10c). ``in_last_t_ho_steps`` is an auxiliary
        OR indicator for the "RLF running in the last T_HO steps" condition
        used later in Eqs. (14d) and (15a).
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # RLF start (linear AND)
        m.addConstr(v.rlf.start[0] == 0, name="rlf_start_0")
        for t in range(1, n_steps):
            m.addConstr(
                v.rlf.start[t] <= v.t310.tau_eq_1[t - 1],
                name=f"rlf_start_ub1_{t}",
            )
            m.addConstr(
                v.rlf.start[t] <= 1 - v.rlf.running[t - 1],
                name=f"rlf_start_ub2_{t}",
            )
            m.addConstr(
                v.rlf.start[t] >= v.t310.tau_eq_1[t - 1] - v.rlf.running[t - 1],
                name=f"rlf_start_lb_{t}",
            )

        # RLF running: OR of starts in last t_rlfr steps (preserving your "can be longer" behavior)
        m.addConstr(v.rlf.running[0] == 0, name="rlf_run_0")
        for t in range(1, n_steps):
            tau_max = min(t, c.t_rlfr_sim - 1)
            sum_starts = v.rlf.start[t - tau_max : t + 1].sum()

            # Avoid division: (tau_max+1) * rlf_running >= sum_starts
            m.addConstr(
                (tau_max + 1) * v.rlf.running[t] >= sum_starts,
                name=f"rlf_run_lb_{t}",
            )
            m.addConstr(
                v.rlf.running[t] <= sum_starts + v.rlf.running[t - 1],
                name=f"rlf_run_ub_{t}",
            )

        # i_rlf_in_last_t_ho_steps: any rlf_running within last t_ho_const steps (including t)
        for t in range(n_steps):
            tau_max = min(t, c.t_ho_sim)
            sum_rlfr = v.rlf.running[t - tau_max : t + 1].sum()
            m.addConstr(
                sum_rlfr >= v.rlf.in_last_t_ho_steps[t], name=f"rlf_prev_lb_{t}"
            )
            m.addConstr(
                sum_rlfr <= (tau_max + 1) * v.rlf.in_last_t_ho_steps[t],
                name=f"rlf_prev_ub_{t}",
            )

    def _add_handover_constraints(self) -> None:
        """Add handover execution constraints.

        Implements Eqs. (14a)-(14d) and (15a)-(15b). The scalar product
        ``x[t-1]^T x[t]`` in Eq. (14a) is represented by per-cell
        ``same_cell`` auxiliaries; max/OR conditions are represented by
        linear inequalities. The initial PCell is free for this genie-aided
        benchmark, but an implementation-level initial hold prevents PCell
        changes during the first ``T_HO`` steps.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # Choose the initial PCell freely, but do not allow an immediate PCell
        # change during the first T_HO steps (no PCell change possible without
        # HO preparation and execution).
        for t in range(1, min(c.t_ho_sim, n_steps)):
            m.addConstr(
                v.x_bs[:, t] == v.x_bs[:, 0],
                name=f"pcell_initial_hold_{t}",
            )

        # PCell change. Written as whole-array constraints rather than a
        # per-(cell, step) loop: gurobipy builds a scipy sparse matrix for
        # every MVar expression, so scalar element access dominated model
        # construction (51% of build time, this block alone 46%). The
        # constraint system is unchanged -- same rows, same coefficients,
        # same order -- only the number of API calls differs.
        m.addConstr(v.ho.same_cell[:, 0] == v.x_bs[:, 0], name="same0")
        if n_steps > 1:
            m.addConstr(v.ho.same_cell[:, 1:] <= v.x_bs[:, :-1], name="same_ub1")
            m.addConstr(v.ho.same_cell[:, 1:] <= v.x_bs[:, 1:], name="same_ub2")
            m.addConstr(
                v.ho.same_cell[:, 1:] >= v.x_bs[:, :-1] + v.x_bs[:, 1:] - 1,
                name="same_lb",
            )

        m.addConstr(
            v.ho.cell_change == 1 - v.ho.same_cell.sum(axis=0),
            name="pcell_change",
        )

        # Rolling window of variable length; kept as a loop because the
        # window shrinks at the trace end.
        for i in range(n_steps):
            tau_max = min(c.t_ho_sim, n_steps - 1 - i)
            m.addConstr(
                v.ho.cell_change[i : i + tau_max + 1].sum() <= 1,
                name=f"pcell_freq_{i}",
            )

        if n_steps > 1:
            m.addConstr(
                v.ho.cell_change[1:] + v.t310.active[:-1] <= 1,
                name="pcell_no_t310",
            )
        m.addConstr(v.ho.cell_change + v.rlf.running <= 1, name="pcell_no_rlf")
        m.addConstr(v.ho.cell_change + v.i_q_out <= 1, name="pcell_no_qout")
        if n_steps > 1:
            m.addConstr(
                v.ho.cell_change[1:]
                <= v.rlf.running[:-1] - v.rlf.in_last_t_ho_steps[1:] + 1,
                name="pcell_after_rlf",
            )

        # HO execution
        m.addConstr(v.ho.exec_end <= v.ho.cell_change, name="hoend_ub1")
        m.addConstr(v.ho.exec_end <= 1 - v.rlf.in_last_t_ho_steps, name="hoend_ub2")
        m.addConstr(
            v.ho.exec_end >= v.ho.cell_change - v.rlf.in_last_t_ho_steps,
            name="hoend_lb",
        )

        self._add_handover_target_constraints()

        # exec_running[t] >= exec_end[t+tau] for every tau in 1..T_exec.
        # Sliced by tau rather than by t, turning n_steps*T_exec scalar
        # constraints into T_exec array constraints covering exactly the
        # same (t, tau) pairs.
        for tau in range(1, c.t_ho_exec_sim + 1):
            if tau >= n_steps:
                break
            m.addConstr(
                v.ho.exec_running[: n_steps - tau] >= v.ho.exec_end[tau:],
                name=f"horun_lb_tau{tau}",
            )
        for t in range(n_steps):
            tau_max = min(c.t_ho_exec_sim, n_steps - t - 1)
            if tau_max == 0:
                m.addConstr(v.ho.exec_running[t] == 0, name=f"horun0_{t}")
                continue
            m.addConstr(
                v.ho.exec_running[t] <= v.ho.exec_end[t + 1 : t + tau_max + 1].sum(),
                name=f"horun_ub_{t}",
            )

    def _add_handover_target_constraints(self) -> None:
        """Restrict regular HOs to the decision-time rank-one neighbor."""
        mode = self.ctx.handover_target_mode
        if mode == "free":
            return
        if mode != "rank1_neighbor":
            raise ValueError(f"unknown handover target mode: {mode}")

        d = self.ctx.data
        if self.n_bs < 2:
            return
        if d.rank1_neighbor is None:
            raise RuntimeError("rank1_neighbor table was not prepared")
        expected_shape = (self.n_bs, self.n_steps)
        if d.rank1_neighbor.shape != expected_shape:
            raise ValueError(
                "rank1_neighbor table must have shape "
                f"{expected_shape}; got {d.rank1_neighbor.shape}"
            )

        delay = self.ctx.consts.t_ho_sim
        n_decisions = max(self.n_steps - delay, 0)
        if n_decisions == 0:
            return
        decisions = np.arange(n_decisions)
        completions = decisions + delay
        v = self.ctx.vars
        for source in range(self.n_bs):
            targets = d.rank1_neighbor[source, decisions]
            self.ctx.model.addConstr(
                v.x_bs[targets, completions]
                >= v.x_bs[source, decisions] + v.ho.exec_end[completions] - 1,
                name=f"ho_rank1_target_source_{source}",
            )

    def _add_rate_constraints(self) -> None:
        """Add rate (capacity) constraints.

        Implements the rate/outage coupling used by Eqs. (1), (2), and (4).
        ``s_rate`` is the selected-cell spectral efficiency, ``conn_aux`` is
        ``1 - i_out``, and ``r_rate`` is the delivered rate after HO/RLF
        interruption gating.
        """
        d = self.ctx.data
        m = self.ctx.model
        v = self.ctx.vars

        # Connectivity: conn_aux = NOT exec_running AND NOT rlf_running.
        m.addConstr(v.rate.conn_aux <= 1 - v.ho.exec_running, name="bconn_ub1")
        m.addConstr(v.rate.conn_aux <= 1 - v.rlf.running, name="bconn_ub2")
        m.addConstr(
            v.rate.conn_aux >= 1 - v.ho.exec_running - v.rlf.running,
            name="bconn_lb",
        )

        # Rate aggregation: s_rate[t] = sum_n cap[n,t] * x[n,t], then the
        # big-M gate that zeroes it during an interruption. cap_max_t is a
        # per-step constant, so the gate is an elementwise product.
        cap_max_t = np.max(d.capacity, axis=0)

        m.addConstr(
            v.rate.s_rate == (d.capacity * v.x_bs).sum(axis=0), name="srate_def"
        )
        m.addConstr(v.rate.r_rate <= v.rate.s_rate, name="rrate_ub_s")
        m.addConstr(v.rate.r_rate <= cap_max_t * v.rate.conn_aux, name="rrate_ub_b")
        m.addConstr(
            v.rate.r_rate >= v.rate.s_rate - cap_max_t * (1 - v.rate.conn_aux),
            name="rrate_lb",
        )

    @property
    def constraints_added(self) -> bool:
        """Flag indicating whether constraints have been added to the model."""
        return self._constraints_added
