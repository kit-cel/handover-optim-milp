"""Plot mean rate and connected time versus lambda from the published dataset."""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .colors import BLUE, RED
from . import utils as ut

matplotlib.use("Agg")

XLIM = (0.1, 1000.0)
YLIM_L_RATE = (5.68, 5.86)
YLIM_R_CONN = (0.956, 0.992)
YTICKS_R = (0.96, 0.97, 0.98, 0.99)


def _print_curve_values(name: str, curve: ut.Curve) -> None:
    """Print curve values as CSV-style rows."""
    print(f"{name} points (lambda,rel_conn,avg_rate):")
    print("lambda,rel_conn,avg_rate")
    for lam, conn, rate in zip(curve.lam, curve.x_mean, curve.y_mean):
        print(f"{lam},{conn:.5f},{rate:.5f}")


def plot_rate_connectivity_tradeoff(
    optim_path: str,
    reference_path: str,
    out_path: str,
    print_values: bool = True,
) -> None:
    """Create tradeoff plot matching the target paper style."""
    opt_curve = ut.load_optimization_curve(optim_path)
    ref_agg = ut.load_reference_parameter_aggregation(reference_path)
    ref_q99 = ut.reference_curve_for_quantile_top_tail(ref_agg, opt_curve.lam, 99.0)

    fig, ax_left = plt.subplots(figsize=(6.6, 5.0))
    ax_right = ax_left.twinx()

    line_handles: list[Line2D] = []

    # Left axis: rate
    line_handles.append(
        ax_left.plot(
            opt_curve.lam,
            opt_curve.y_mean,
            color=BLUE,
            linewidth=2.0,
            linestyle="-",
            marker="s",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            label=r"$\mathcal{L}_{\text{r}}(\lambda)$",
        )[0]
    )

    line_handles.append(
        ax_left.plot(
            ref_q99.lam,
            ref_q99.y_mean,
            color=RED,
            linewidth=2.0,
            linestyle="-",
            marker="^",
            markersize=5.8,
            markerfacecolor="white",
            markeredgecolor=RED,
            label=r"$\bar{L}_{\text{r,99}}(\lambda)$",
        )[0]
    )

    # Right axis: connected time
    line_handles.append(
        ax_right.plot(
            opt_curve.lam,
            opt_curve.x_mean,
            color=BLUE,
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            label=r"$\mathcal{L}_{\text{c}}(\lambda)$",
        )[0]
    )

    line_handles.append(
        ax_right.plot(
            ref_q99.lam,
            ref_q99.x_mean,
            color=RED,
            linewidth=2.0,
            linestyle="--",
            marker="^",
            markersize=5.8,
            markerfacecolor="white",
            markeredgecolor=RED,
            label=r"$\bar{L}_{\text{c,99}}(\lambda)$",
        )[0]
    )

    ax_left.set_xscale("log")

    ax_left.set_xlim(XLIM)
    ax_left.set_ylim(YLIM_L_RATE)
    ax_right.set_ylim(YLIM_R_CONN)
    ax_right.set_yticks(YTICKS_R)

    ax_left.set_xlabel(r"$\lambda$", fontsize=10)
    ax_left.set_ylabel("Average achieved rate (bit/s/Hz)", fontsize=10)
    ax_right.set_ylabel("Relative connected time", fontsize=10)

    ax_left.grid(True, which="major", linewidth=0.5, alpha=0.6)

    ax_left.tick_params(labelsize=9)
    ax_right.tick_params(labelsize=9)

    ax_right.legend(
        handles=line_handles,
        labels=[str(h.get_label()) for h in line_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    if print_values:
        _print_curve_values("Optimization", opt_curve)
        _print_curve_values("Reference q=99", ref_q99)

    fig.tight_layout()

    full_out_path = os.path.join(out_path, "rate_connected_time_vs_lambda_q99.png")
    fig.savefig(full_out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {full_out_path}")
