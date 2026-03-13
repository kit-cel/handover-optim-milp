"""Plot mean rate and connected time versus lambda from the published dataset."""

import matplotlib
import matplotlib.pyplot as plt

from .utils import Curve

matplotlib.use("Agg")

BLUE = "#0072B2"
RED = "#D55E00"


def plot_rate_connectivity_tradeoff(
    opt: Curve,
    ref_q99: Curve,
    out_path: str,
    print_values: bool = True,
) -> None:
    """Create tradeoff plot matching the target paper style."""
    fig, ax_left = plt.subplots(figsize=(6.6, 5.0))
    ax_right = ax_left.twinx()

    # Left axis: rate
    opt_rate = ax_left.plot(
        opt.lam,
        opt.y_mean,
        color=BLUE,
        linewidth=2.0,
        linestyle="-",
        marker="s",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        label=r"$\mathcal{L}_{\text{r}}(\lambda)$",
    )[0]

    ref_rate = ax_left.plot(
        ref_q99.lam,
        ref_q99.y_mean,
        color=RED,
        linewidth=2.0,
        linestyle="-",
        marker="^",
        markersize=5.8,
        markerfacecolor="white",
        markeredgecolor=RED,
        label=r"$\bar L_{\text{r,99}}(\lambda)$",
    )[0]

    # Right axis: connected time
    opt_conn = ax_right.plot(
        opt.lam,
        opt.x_mean,
        color=BLUE,
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        label=r"$\mathcal{L}_{\text{c}}(\lambda)$",
    )[0]

    ref_conn = ax_right.plot(
        ref_q99.lam,
        ref_q99.x_mean,
        color=RED,
        linewidth=2.0,
        linestyle="--",
        marker="^",
        markersize=5.8,
        markerfacecolor="white",
        markeredgecolor=RED,
        label=r"$\bar L_{\text{c,99}}(\lambda)$",
    )[0]

    ax_left.set_xscale("log")
    ax_left.set_xlim(0.1, 1000.0)

    ax_left.set_xlabel(r"$\lambda$", fontsize=10)
    ax_left.set_ylabel("Average achieved rate (bit/s/Hz)", fontsize=10)
    ax_right.set_ylabel("Relative connected time", fontsize=10)

    ax_left.set_ylim(5.68, 5.86)
    ax_right.set_ylim(0.956, 0.992)
    ax_right.set_yticks([0.96, 0.97, 0.98, 0.99])

    ax_left.grid(True, which="major", linewidth=0.5, alpha=0.6)

    ax_left.tick_params(labelsize=9)
    ax_right.tick_params(labelsize=9)

    legend_handles = [opt_rate, ref_rate, opt_conn, ref_conn]
    legend_labels = [str(h.get_label()) for h in legend_handles]
    ax_right.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    if print_values:
        print("Optimization points (lambda,avg_rate,rel_conn):")
        print("lambda,avg_rate,rel_conn")
        for lam, rate, conn in zip(opt.lam, opt.y_mean, opt.x_mean):
            print(f"{lam},{rate:.5f},{conn:.5f}")

        print("Reference q=99 points (lambda,avg_rate,rel_conn):")
        print("lambda,avg_rate,rel_conn")
        for lam, rate, conn in zip(ref_q99.lam, ref_q99.y_mean, ref_q99.x_mean):
            print(f"{lam},{rate:.5f},{conn:.5f}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
