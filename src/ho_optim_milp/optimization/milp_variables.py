"""MILP variable definitions for the HO optimization problem."""

from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True)
class CounterVars:
    """
    MILP formulation of a threshold counter (e.g., N310/N311).

    Per time step t:
    u_raw[t] : int
        raw counter update
    u_sat[t] : int
        saturated counter
    cnt[t] : int
        stored counter
    ind[t] : bool
        threshold indicator

    Binary auxiliaries:
    -------------------
    b_q[t] : bool
        increment event (gated input, e.g., q_out)
    b_reset[t] : bool
        counter enabled (e.g., no RLF / no HO)
    b_c[t] : bool
        counter retained (e.g., enabled AND no active timer)
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
    """
    T310 timer (RLF detection) related MILP variables.

    Per time step t:
    start[t] : bool
        Timer start event (triggered when N310 threshold is reached).
    stop[t] : bool
        Timer stop event (e.g., due to N311 or other recovery conditions).
    active[t] : bool
        Timer active state (1 while T310 is running).
    cancel_due_to_n311[t] : bool
        Indicator that T310 is stopped due to N311 condition.
    expiry[t] : bool
        Timer expiry event (1 when T310 reaches its maximum duration -> RLF).
    tau[t] : int
        Remaining timer value (counts down while active).
    tau_eq_1[t] : bool
        Indicator that tau[t] = 1 (used to detect expiry at next step).

    Binary auxiliaries (mode selection):
    ------------------------------------
    b_mode_rlf[t]   : RLF mode (timer expired).
    b_mode_start[t] : Start mode (timer initialized).
    b_mode_stop[t]  : Stop mode (timer reset to inactive).
    b_mode_decr[t]  : Decrement mode (timer counts down).
    """

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
    """
    Radio Link Failure (RLF) related MILP variables.

    Per time step t:
    start[t] : bool
        RLF start indicator (1 iff an RLF is triggered at t).
    running[t] : bool
        RLF active state (1 while the RLF recovery procedure is ongoing).
    in_last_t_ho_steps[t] : bool
        Indicator that a handover occurred within the last T_HO = T_prep + T_exec steps.
    """

    start: gp.MVar
    running: gp.MVar
    in_last_t_ho_steps: gp.MVar


@dataclass(frozen=True)
class HandoverVars:
    """
    Handover-related MILP variables.

    Per time step t:
    same_cell[t] : bool
        Indicator that the serving cell remains unchanged from t-1 to t.
    cell_change[t] : bool
        Indicator that a handover decision occurs at t, i.e., serving cell at t
        differs from the serving cell at t-1.
    exec_running[t] : bool
        HO execution active (1 during the HO execution period).
    exec_end[t] : bool
        Indicator that HO execution completes at t, i.e., transition from
        exec_running[t-1]=1 to exec_running[t]=0.
    """

    same_cell: gp.MVar
    cell_change: gp.MVar
    exec_end: gp.MVar
    exec_running: gp.MVar


@dataclass(frozen=True)
class RateVars:
    """
    Variables related to rate and connectivity.


    Per time step t:
    conn_aux[t] : bool
        Connectivity indicator (1 if UE is connected / no outage, 0 otherwise).
    s_rate[t] : float
        Instantaneous spectral efficiency (unconstrained / pre-gating rate).
    r_rate[t] : float
        Effective (realized/achieved) rate: r_rate[t] = conn_aux[t] * s_rate[t]
    """

    conn_aux: gp.MVar
    s_rate: gp.MVar
    r_rate: gp.MVar


@dataclass(frozen=True)
class ModelVars:
    """
    Collection of all MILP decision and state variables.

    Per time step t:
    x_bs[b, t] : bool
        Serving cell assignment (1 if cell b is selected at t).
    i_q_in[t], i_q_out[t] : bool
        In-sync / out-of-sync link quality indicators.

    n310, n311 : CounterVars
        Threshold counters for RLF triggering and recovery.
    t310 : T310Vars
        Timer state associated with N310 (RLF detection).
    rlf : RLFVars
        Radio Link Failure state (start, running, gating conditions).
    ho : HandoverVars
        Handover decision and execution state.
    rate : RateVars
        Connectivity and rate variables.
    """

    x_bs: gp.MVar
    i_q_in: gp.MVar
    i_q_out: gp.MVar
    n310: CounterVars
    n311: CounterVars
    t310: T310Vars
    rlf: RLFVars
    ho: HandoverVars
    rate: RateVars
