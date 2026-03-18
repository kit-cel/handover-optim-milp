"""Plot"""

import argparse

from ho_optim_milp.plotting import plot_rate_connectivity_tradeoff
from ho_optim_milp.plotting import utils as ut

DATASET_DIR: str = "dataset_root"


def add_parser(subparsers: argparse._SubParsersAction, cwd: str) -> None:
    """Argument parser for main function."""
    parser: argparse.ArgumentParser = subparsers.add_parser("plot_tradeoff")

    parser.add_argument(
        "--optim-path",
        dest="optim_path",
        default=ut.get_default_optim_result_path(cwd, DATASET_DIR),
        help="Path to optim_result_metrics.parquet",
    )
    parser.add_argument(
        "--reference-path",
        dest="reference_path",
        default=ut.get_default_reference_result_path(cwd, DATASET_DIR),
        help="Path to reference_result_metrics.parquet",
    )
    parser.add_argument(
        "--out-path",
        dest="out_path",
        default=ut.get_default_plot_path(cwd),
        help="Output path for figures.",
    )
    parser.add_argument(
        "--print-values",
        dest="print_values",
        action="store_true",
        help="Print plotted values to stdout.",
    )

    parser.set_defaults(func=main)


def main(optim_path: str, reference_path: str, out_path: str, **kwargs) -> int:
    """Load published datasets and plot the tradeoff versus lambda."""
    plot_rate_connectivity_tradeoff(
        optim_path=optim_path,
        reference_path=reference_path,
        out_path=out_path,
        print_values=kwargs.get("print_values", False),
    )

    return 0
