"""Optimization utility functions."""

from typing import TypeAlias, cast

import gurobipy as gp
from gurobipy import GRB

GPVar: TypeAlias = gp.Var | gp.MVar
GPScalar: TypeAlias = gp.Var | gp.MVar


def _as_var(x: GPScalar) -> gp.Var:
    """Extract a scalar ``gp.Var`` from a ``gp.MVar``; raises if the MVar is not size-1."""
    if isinstance(x, gp.MVar):
        if x.size != 1:
            raise ValueError(f"Expected scalar MVar, got size={x.size}")
        return cast(gp.Var, x.item())
    return x


def lin_binary_and(
    model: gp.Model, z: GPScalar, a: GPScalar, b: GPScalar, name: str
) -> None:
    """Enforce z = a AND b for binary variables a,b,z."""
    z_var = _as_var(z)
    a_var = _as_var(a)
    b_var = _as_var(b)
    model.addConstr(z_var <= a_var, name=f"{name}_ub_a")
    model.addConstr(z_var <= b_var, name=f"{name}_ub_b")
    model.addConstr(z_var >= a_var + b_var - 1, name=f"{name}_lb")


def lin_binary_or(model: gp.Model, z: GPScalar, xs: list[GPScalar], name: str) -> None:
    """Enforce z = OR of list of binary variables xs."""
    z_var = _as_var(z)
    xs_vars = [_as_var(x) for x in xs]
    for i, x in enumerate(xs_vars):
        model.addConstr(z_var >= x, name=f"{name}_lb_{i}")
    model.addConstr(z_var <= gp.quicksum(xs_vars), name=f"{name}_ub_sum")


def lin_prod_bin_int(
    model: gp.Model,
    y: GPScalar,
    b: GPScalar,
    x: GPScalar,
    x_ub: int,
    *,
    name: str | None,
) -> None:
    """Enforce y = b * x for binary b and integer x in [0, x_ub]."""
    y_var = _as_var(y)
    b_var = _as_var(b)
    x_var = _as_var(x)
    name = "constr" if name is None else name

    model.addConstr(y_var <= x_var, name=f"{name}_ub_x")
    model.addConstr(y_var <= x_ub * b_var, name=f"{name}_ub_b")
    model.addConstr(y_var >= x_var - x_ub * (1 - b_var), name=f"{name}_lb")
    model.addConstr(y_var >= 0, name=f"{name}_lb0")


def lin_saturate_min(
    model: gp.Model,
    u_sat: GPScalar,
    u_raw: GPScalar,
    u_sat_ub: int,
    u_raw_ub: int,
    *,
    name: str | None = None,
) -> gp.Var:
    """Linearized saturation function."""
    u_sat_var = _as_var(u_sat)
    u_raw_var = _as_var(u_raw)
    name = "constr" if name is None else name

    b_overflow = model.addVar(vtype=GRB.BINARY, name=f"{name}_over")

    # b_overflow=0 => u_raw <= u_sat_ub-1
    model.addConstr(
        u_raw_var <= (u_sat_ub - 1) + (u_raw_ub - (u_sat_ub - 1)) * b_overflow,
        name=f"{name}_or_ub",
    )
    # b_overflow=1 => u_raw >= u_sat_ub
    model.addConstr(u_raw_var >= u_sat_ub * b_overflow, name=f"{name}_or_lb")

    # u_sat = min(u_raw, u_sat_ub)
    model.addConstr(u_sat_var <= u_raw_var, name=f"{name}_sat_ub_raw")
    model.addConstr(u_sat_var <= u_sat_ub, name=f"{name}_sat_ub_N")

    big_m_u_raw = u_raw_ub
    model.addConstr(
        u_sat_var >= u_raw_var - big_m_u_raw * b_overflow,
        name=f"{name}_sat_lb_raw_if_o0",
    )
    model.addConstr(
        u_sat_var >= u_sat_ub - u_sat_ub * (1 - b_overflow),
        name=f"{name}_sat_lb_N_if_o1",
    )
    model.addConstr(
        u_sat_var <= u_raw_var + big_m_u_raw * b_overflow,
        name=f"{name}_sat_ub_raw_if_o0",
    )
    model.addConstr(
        u_sat_var <= u_sat_ub + u_sat_ub * (1 - b_overflow),
        name=f"{name}_sat_ub_N_if_o1",
    )

    return b_overflow
