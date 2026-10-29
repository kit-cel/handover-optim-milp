"""MILP variable definitions for the HO optimization problem."""

from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True)
class CounterVars:
    """N310/N311 counter state and linearization auxiliaries.

    ``u_raw`` is the update, ``u_sat`` its clipped value, and ``cnt`` the
    stored count. ``b_q``, ``b_reset``, and ``b_c`` gate its transitions.
    """

    u_raw: gp.MVar
    u_sat: gp.MVar
    cnt: gp.MVar
    ind: gp.MVar
    b_reset: gp.MVar
    b_q: gp.MVar
    b_c: gp.MVar


@dataclass(frozen=True)
class T310Vars:
    """T310 state and transition modes; ``tau`` counts remaining steps."""

    start: gp.MVar
    stop: gp.MVar
    active: gp.MVar
    cancel_due_to_n311: gp.MVar
    expiry: gp.MVar
    tau: gp.MVar
    tau_eq_1: gp.MVar
    b_mode_rlf: gp.MVar
    b_mode_start: gp.MVar
    b_mode_stop: gp.MVar
    b_mode_decr: gp.MVar


@dataclass(frozen=True)
class RLFVars:
    """RLF start, recovery state, and recent-handover flag."""

    start: gp.MVar
    running: gp.MVar
    in_last_t_ho_steps: gp.MVar


@dataclass(frozen=True)
class HandoverVars:
    """Handover decision, execution state, and completion flag."""

    same_cell: gp.MVar
    cell_change: gp.MVar
    exec_end: gp.MVar
    exec_running: gp.MVar


@dataclass(frozen=True)
class RateVars:
    """Connectivity and rate before (``s_rate``) and after outage gating."""

    conn_aux: gp.MVar
    s_rate: gp.MVar
    r_rate: gp.MVar


@dataclass(frozen=True)
class ModelVars:
    """All MILP decision and state variables."""

    x_bs: gp.MVar
    i_q_in: gp.MVar
    i_q_out: gp.MVar
    n310: CounterVars
    n311: CounterVars
    t310: T310Vars
    rlf: RLFVars
    ho: HandoverVars
    rate: RateVars
