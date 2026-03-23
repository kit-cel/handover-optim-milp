"""Script to plot the rate-connectivity trade-off."""

from argparse import _SubParsersAction as SubParsersAct

from ho_optim_milp.common.subparsers import add_default_plot_parser
from ho_optim_milp.plotting import plot_rate_connectivity_tradeoff


def add_parser(
    subparsers: SubParsersAct, cwd: str, name: str = "plot_tradeoff"
) -> None:
    """Argument parser for main function."""
    add_default_plot_parser(subparsers, main, cwd, name)


def main(optim_path: str, reference_path: str, out_path: str, **kwargs) -> int:
    """Load published datasets and plot the tradeoff versus lambda."""
    plot_rate_connectivity_tradeoff(
        optim_path=optim_path,
        reference_path=reference_path,
        out_path=out_path,
        print_values=kwargs.get("print_values", False),
    )

    return 0
