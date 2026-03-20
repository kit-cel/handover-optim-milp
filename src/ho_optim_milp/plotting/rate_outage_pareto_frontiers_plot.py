"""Plot rate-outage Pareto frontiers from the published dataset."""

import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .colors import BLUE, GREEN, PURPLE, RED
from . import utils as ut


matplotlib.use("Agg")

REF_PERCENTILES: list[float] = [90.0, 95.0, 99.0]

STYLE_MAP = {
    99.0: {"color": RED, "marker": "^"},
    95.0: {"color": GREEN, "marker": "o"},
    90.0: {"color": PURPLE, "marker": "D"},
    "default": {"color": "black", "marker": "o"},
}


def _build_file_path(out_path: str) -> str:
    pct_tag = "q-" + "-".join(f"{q:g}" for q in REF_PERCENTILES)
    full_out_path = os.path.join(
        out_path, f"rate_outage_pareto_frontiers_{pct_tag}.png"
    )
    return full_out_path


def _find_point_for_lambda(
    curve: ut.Curve,
    lambda_value: float,
    atol: float = 1e-12,
) -> tuple[float, float] | None:
    """Return (outage, rate) point for a given lambda if present."""
    idx = np.where(np.isclose(curve.lam, lambda_value, rtol=0.0, atol=atol))[0]
    if idx.size == 0:
        return None

    i = int(idx[0])
    outage = 100.0 * (1.0 - curve.x_mean[i])
    rate = curve.y_mean[i]
    return float(outage), float(rate)


def _get_color(curve: ut.Curve) -> Any:
    pct = float(curve.pct) if curve.pct is not None else None
    style = STYLE_MAP.get(pct, STYLE_MAP["default"])  # type: ignore
    return style["color"]


def _get_marker(curve: ut.Curve) -> str:
    pct = float(curve.pct) if curve.pct is not None else None
    style = STYLE_MAP.get(pct, STYLE_MAP["default"])  # type: ignore
    return style["marker"]


def _annotate_lambdas(ax: Axes, opt_curve: ut.Curve) -> None:
    p_low = _find_point_for_lambda(opt_curve, 0.1)
    if p_low is not None:
        ax.annotate(
            r"$\lambda=0.1$",
            xy=p_low,
            xytext=(2.75, 5.84),
            textcoords="data",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )

    p_high = _find_point_for_lambda(opt_curve, 1000.0)
    if p_high is not None:
        ax.annotate(
            r"$\lambda=1000$",
            xy=p_high,
            xytext=(2.3, 5.72),
            textcoords="data",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        )


def _print_curve_values(name: str, curve: ut.Curve) -> None:
    """Print curve values as CSV-style rows."""
    print(f"{name} points (lambda,outage,avg_rate):")
    print("lambda,outage,avg_rate")
    for lam, conn, rate in zip(curve.lam, curve.x_mean, curve.y_mean):
        out = 100.0 * (1.0 - conn)
        print(f"{lam},{out:.5f},{rate:.5f}")


def plot_pareto_fronts(
    optim_path: str,
    reference_path: str,
    out_path: str,
    print_values: bool = False,
    annotate_lambdas: bool = True,
) -> None:
    """Create Pareto scatter plot matching the target paper style."""
    opt_curve = ut.load_optimization_curve(optim_path)
    ref_agg = ut.load_reference_parameter_aggregation(reference_path)
    bound_conn, bound_cap = ut.reference_pareto_bound(ref_agg)

    ref_curves: list[ut.Curve] = [
        ut.reference_curve_for_quantile_top_tail(ref_agg, opt_curve.lam, q)
        for q in REF_PERCENTILES
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    # MILP / optimization Pareto front
    ax.plot(
        100.0 * (1.0 - opt_curve.x_mean),
        opt_curve.y_mean,
        color=BLUE,
        linewidth=2.0,
        linestyle="-",
        marker="s",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        label="MILP Pareto front",
    )
    if annotate_lambdas:
        _annotate_lambdas(ax, opt_curve)

    # Reference Pareto bound
    ax.plot(
        100.0 * (1.0 - bound_conn),
        bound_cap,
        color=RED,
        linewidth=2.0,
        linestyle=":",
        label="Reference frontier",
    )

    # Reference top-tail averages
    for curve in ref_curves:
        ax.plot(
            100.0 * (1.0 - curve.x_mean),
            curve.y_mean,
            color=_get_color(curve),
            linewidth=1.5,
            linestyle="-",
            marker=_get_marker(curve),
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=_get_color(curve),
            label=(
                f"Reference ($q={int(curve.pct)}$)"
                if curve.pct is not None
                else "Reference"
            ),
        )

    ax.set_xlabel(r"Outage $1-\mathcal{L}_{c}$ (%)", fontsize=10)
    ax.set_ylabel("Average achieved rate (bit/s/Hz)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.6)

    ax.set_xlim(1.0, 5.25)
    ax.set_ylim(5.68, 5.86)
    ax.invert_xaxis()

    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white")

    if print_values:
        _print_curve_values("Optimization", opt_curve)
        for curve in ref_curves:
            _print_curve_values(f"Reference q={curve.pct:g}", curve)

        print("Reference Pareto bound points (outage,avg_rate):")
        print("outage,avg_rate")
        for conn, rate in zip(bound_conn, bound_cap):
            print(f"{100.0 * (1.0 - conn):.5f},{rate:.5f}")

    fig.tight_layout()
    fig.savefig(_build_file_path(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")
