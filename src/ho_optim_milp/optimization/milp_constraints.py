"""Module defining all constraints for the handover optimization MILP model."""

from gurobipy import GRB
import numpy as np

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

        Enforces a one-hot cell assignment per time step, defines the
        Q-in/Q-out binary indicators, and optionally fixes ``x_bs`` to
        the reference simulation labels in debug mode.
        """
        n_steps = self.n_steps
        d = self.ctx.data
        m = self.ctx.model
        v = self.ctx.vars

        # one-hot PCell
        m.addConstr(v.x_bs.sum(axis=0) == 1, name="pcell_onehot")

        # Q_in/Q_out indicators
        for t in range(n_steps):
            m.addConstr(
                v.i_q_in[t] == (d.q_in_mat[:, t] * v.x_bs[:, t]).sum(),
                name=f"q_in_def_{t}",
            )
            m.addConstr(
                v.i_q_out[t] == (d.q_out_mat[:, t] * v.x_bs[:, t]).sum(),
                name=f"q_out_def_{t}",
            )

        # fixed PCells for debugging
        if self.ctx.debug is True:
            if self.ctx.ref_results is None:
                raise RuntimeError("Reference results not available for debugging.")
            m.addConstr(
                v.x_bs
                == np.eye(self.n_bs, dtype=int)[:, self.ctx.ref_results["pcell"]],
                name="debug_fixed_x",
            )

    def _add_n310_constraints(self) -> None:
        """Add N310 out-of-sync counter constraints.

        Models the N310 counter that increments on Q-out indications and
        resets on Q-in; triggers T310 when the counter saturates.
        """
        n_steps = self.n_steps
        m = self.ctx.model
        v = self.ctx.vars
        c = self.ctx.consts

        # N310
        for t in range(n_steps):
            m.addConstr(
                v.n310.b_reset[t] <= 1 - v.rlf.running[t],
                name=f"reset310_ub1_{t}",
            )
            m.addConstr(
                v.n310.b_reset[t] <= 1 - v.ho.exec_running[t],
                name=f"reset310_ub2_{t}",
            )
            m.addConstr(
                v.n310.b_reset[t] >= 1 - v.rlf.running[t] - v.ho.exec_running[t],
                name=f"reset310_lb_{t}",
            )

            m.addConstr(v.n310.b_c[t] <= v.n310.b_reset[t], name=f"c310_ub1_{t}")
            m.addConstr(
                v.n310.b_c[t] <= 1 - v.t310.active[t],
                name=f"c310_ub2_{t}",
            )
            m.addConstr(
                v.n310.b_c[t] >= v.n310.b_reset[t] - v.t310.active[t],
                name=f"c310_lb_{t}",
            )

        m.addConstr(v.n310.b_q[0] == v.i_q_out[0], name="bq310_0")
        for t in range(1, n_steps):
            m.addConstr(v.n310.b_q[t] <= v.i_q_out[t], name=f"bq310_ub1_{t}")
            m.addConstr(
                v.n310.b_q[t] <= 1 - v.t310.active[t - 1],
                name=f"bq310_ub2_{t}",
            )
            m.addConstr(
                v.n310.b_q[t] >= v.i_q_out[t] - v.t310.active[t - 1],
                name=f"bq310_lb_{t}",
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

        ub_u_n310 = c.n310 + 1
        for t in range(n_steps):
            m.addConstr(
                v.n310.u_raw[t] >= c.n310 * v.n310.ind[t],
                name=f"i310_lb_{t}",
            )
            m.addConstr(
                v.n310.u_raw[t]
                <= (c.n310 - 1) + (ub_u_n310 - (c.n310 - 1)) * v.n310.ind[t],
                name=f"i310_ub_{t}",
            )

    def _add_n311_constraints(self) -> None:
        """Add N311 in-sync counter constraints.

        Models the N311 counter that increments on Q-in indications while
        T310 is active; cancels T310 when the counter saturates.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # N311
        for t in range(n_steps):
            m.addConstr(
                v.n311.b_reset[t] <= 1 - v.rlf.running[t],
                name=f"reset311_ub1_{t}",
            )
            m.addConstr(
                v.n311.b_reset[t] <= 1 - v.ho.exec_running[t],
                name=f"reset311_ub2_{t}",
            )
            m.addConstr(
                v.n311.b_reset[t] >= 1 - v.rlf.running[t] - v.ho.exec_running[t],
                name=f"reset311_lb_{t}",
            )

        m.addConstr(v.n311.b_q[0] == 0, name="bq311_0")
        for t in range(1, n_steps):
            m.addConstr(v.n311.b_q[t] <= v.i_q_in[t], name=f"bq311_ub1_{t}")
            m.addConstr(
                v.n311.b_q[t] <= v.t310.active[t - 1],
                name=f"bq311_ub2_{t}",
            )
            m.addConstr(
                v.n311.b_q[t] >= v.i_q_in[t] + v.t310.active[t - 1] - 1,
                name=f"bq311_lb_{t}",
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

        ub_u_n311 = c.n311 + 1
        for t in range(n_steps):
            m.addConstr(
                v.n311.u_raw[t] >= c.n311 * v.n311.ind[t],
                name=f"i311_lb_{t}",
            )
            m.addConstr(
                v.n311.u_raw[t]
                <= (c.n311 - 1) + (ub_u_n311 - (c.n311 - 1)) * v.n311.ind[t],
                name=f"i311_ub_{t}",
            )

    def _add_t310_constraints(self) -> None:
        """Add T310 timer constraints.

        Implements the T310 countdown timer: active/start/stop transitions,
        remaining-time variable ``tau``, and the ``tau_eq_1`` indicator used
        to detect timer expiry at the end of each step.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # T310: define active and eq1 from integer tau (exact)
        # active: tau >= 1  <=>  tau >= i_active and tau <= t310*i_active
        for t in range(n_steps):
            m.addConstr(
                v.t310.tau[t] >= v.t310.active[t],
                name=f"t310_act_lb_{t}",
            )
            m.addConstr(
                v.t310.tau[t] <= c.t310 * v.t310.active[t],
                name=f"t310_act_ub_{t}",
            )

            # eq1: z=1 <=> tau == 1
            z = v.t310.tau_eq_1[t]
            m.addConstr(
                v.t310.tau[t] - 1 <= (c.t310 - 1) * (1 - z),
                name=f"t310_eq1_ub_{t}",
            )
            m.addConstr(
                1 - v.t310.tau[t] <= 1 * (1 - z),
                name=f"t310_eq1_lb_{t}",
            )
            # tau = 1 => z = 1  (uses 'active' to eliminate the tau=0 case)
            # Equivalent to: if a=1 and z=0 then tau>=2
            if c.t310 >= 2:
                m.addConstr(
                    v.t310.tau[t] >= 2 * (v.t310.active[t] - z),
                    name=f"t310_eq1_rev_{t}",
                )

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

        # Mode selection and tau update
        for t in range(n_steps):
            m.addConstr(
                v.t310.b_mode_rlf[t] == v.rlf.running[t],
                name=f"mrlf_{t}",
            )

            m.addConstr(
                v.t310.b_mode_start[t] <= v.t310.start[t],
                name=f"mstart_ub_{t}",
            )
            m.addConstr(
                v.t310.b_mode_start[t] <= 1 - v.t310.b_mode_rlf[t],
                name=f"mstart_norl_{t}",
            )
            m.addConstr(
                v.t310.b_mode_start[t] >= v.t310.start[t] - v.t310.b_mode_rlf[t],
                name=f"mstart_lb_{t}",
            )

            m.addConstr(
                v.t310.b_mode_stop[t] <= v.t310.stop[t],
                name=f"mstop_ub_{t}",
            )
            m.addConstr(
                v.t310.b_mode_stop[t] <= 1 - v.t310.b_mode_rlf[t],
                name=f"mstop_norl_{t}",
            )
            m.addConstr(
                v.t310.b_mode_stop[t] <= 1 - v.t310.b_mode_start[t],
                name=f"mstop_nostart_{t}",
            )
            m.addConstr(
                v.t310.b_mode_stop[t]
                >= v.t310.stop[t] - v.t310.b_mode_rlf[t] - v.t310.b_mode_start[t],
                name=f"mstop_lb_{t}",
            )

            m.addConstr(
                v.t310.b_mode_decr[t]
                == 1
                - v.t310.b_mode_rlf[t]
                - v.t310.b_mode_start[t]
                - v.t310.b_mode_stop[t],
                name=f"mdecr_{t}",
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

        Implements the RLF start, the running-recovery window, and the
        ``in_last_t_ho_steps`` indicator used to suppress handover decisions
        immediately after an RLF.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # RLF start (linear AND)
        m.addConstr(v.rlf.start[0] == 0, name="rlf_start_0")
        for t in range(1, n_steps):
            if self.ctx.debug is True:
                if self.ctx.ref_results is None:
                    raise RuntimeError("Reference results not available for debugging.")
                ref_hof = self.ctx.ref_results["hof_detected"]
                if ref_hof[t]:
                    m.addConstr(v.rlf.start[t] == 1, name=f"rlf_force_{t}")
                    continue

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

        Defines same-cell, cell-change, and handover-execution variables;
        enforces frequency limits and the causal execution window.
        """
        n_steps = self.n_steps
        c = self.ctx.consts
        m = self.ctx.model
        v = self.ctx.vars

        # PCell change
        for n in range(self.n_bs):
            m.addConstr(v.ho.same_cell[n, 0] == v.x_bs[n, 0], name=f"same0_{n}")
            for t in range(1, n_steps):
                m.addConstr(
                    v.ho.same_cell[n, t] <= v.x_bs[n, t - 1],
                    name=f"same_ub1_{n}_{t}",
                )
                m.addConstr(
                    v.ho.same_cell[n, t] <= v.x_bs[n, t],
                    name=f"same_ub2_{n}_{t}",
                )
                m.addConstr(
                    v.ho.same_cell[n, t] >= v.x_bs[n, t - 1] + v.x_bs[n, t] - 1,
                    name=f"same_lb_{n}_{t}",
                )

        for t in range(n_steps):
            m.addConstr(
                v.ho.cell_change[t] == 1 - v.ho.same_cell[:, t].sum(),
                name=f"pcell_change_{t}",
            )

        for i in range(n_steps - c.t_ho_sim):
            m.addConstr(
                v.ho.cell_change[i : i + c.t_ho_sim + 1].sum() <= 1,
                name=f"pcell_freq_{i}",
            )

        for t in range(1, n_steps):
            m.addConstr(
                v.ho.cell_change[t] + v.t310.active[t - 1] <= 1,
                name=f"pcell_no_t310_{t}",
            )

        for t in range(n_steps):
            m.addConstr(
                v.ho.cell_change[t] + v.rlf.running[t] <= 1,
                name=f"pcell_no_rlf_{t}",
            )
            m.addConstr(
                v.ho.cell_change[t] + v.i_q_out[t] <= 1,
                name=f"pcell_no_qout_{t}",
            )

        for t in range(1, n_steps):
            m.addConstr(
                v.ho.cell_change[t]
                <= v.rlf.running[t - 1] - v.rlf.in_last_t_ho_steps[t] + 1,
                name=f"pcell_after_rlf_{t}",
            )

        # HO execution
        for t in range(n_steps):
            m.addConstr(
                v.ho.exec_end[t] <= v.ho.cell_change[t],
                name=f"hoend_ub1_{t}",
            )
            m.addConstr(
                v.ho.exec_end[t] <= 1 - v.rlf.in_last_t_ho_steps[t],
                name=f"hoend_ub2_{t}",
            )
            m.addConstr(
                v.ho.exec_end[t] >= v.ho.cell_change[t] - v.rlf.in_last_t_ho_steps[t],
                name=f"hoend_lb_{t}",
            )

        for t in range(n_steps):
            tau_max = min(c.t_ho_exec_sim, n_steps - t - 1)
            if tau_max == 0:
                m.addConstr(v.ho.exec_running[t] == 0, name=f"horun0_{t}")
                continue

            for tau in range(1, tau_max + 1):
                m.addConstr(
                    v.ho.exec_running[t] >= v.ho.exec_end[t + tau],
                    name=f"horun_lb_{t}_{tau}",
                )
            m.addConstr(
                v.ho.exec_running[t] <= v.ho.exec_end[t + 1 : t + tau_max + 1].sum(),
                name=f"horun_ub_{t}",
            )

    def _add_rate_constraints(self) -> None:
        """Add rate (capacity) constraints.

        Computes the serving-cell rate via big-M linearisation: zero during
        HO execution or RLF recovery, otherwise equal to the selected cell's
        Shannon capacity.
        """
        n_steps = self.n_steps
        d = self.ctx.data
        m = self.ctx.model
        v = self.ctx.vars

        # Connectivity
        for t in range(n_steps):
            m.addConstr(
                v.rate.conn_aux[t] <= 1 - v.ho.exec_running[t],
                name=f"bconn_ub1_{t}",
            )
            m.addConstr(
                v.rate.conn_aux[t] <= 1 - v.rlf.running[t],
                name=f"bconn_ub2_{t}",
            )
            m.addConstr(
                v.rate.conn_aux[t] >= 1 - v.ho.exec_running[t] - v.rlf.running[t],
                name=f"bconn_lb_{t}",
            )

        # Rate aggregation: s_rate[t] = sum_n cap[n,t] * x[n,t]
        cap_max_t = np.max(d.capacity, axis=0)

        for t in range(n_steps):
            m.addConstr(
                v.rate.s_rate[t] == (d.capacity[:, t] * v.x_bs[:, t]).sum(),
                name=f"srate_def_{t}",
            )
            m.addConstr(
                v.rate.r_rate[t] <= v.rate.s_rate[t],
                name=f"rrate_ub_s_{t}",
            )
            m.addConstr(
                v.rate.r_rate[t] <= cap_max_t[t] * v.rate.conn_aux[t],
                name=f"rrate_ub_b_{t}",
            )
            m.addConstr(
                v.rate.r_rate[t]
                >= v.rate.s_rate[t] - cap_max_t[t] * (1 - v.rate.conn_aux[t]),
                name=f"rrate_lb_{t}",
            )

    @property
    def constraints_added(self) -> bool:
        """Flag indicating whether constraints have been added to the model."""
        return self._constraints_added
