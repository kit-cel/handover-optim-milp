"""Plot average achieved rate-outage Pareto frontiers."""

import argparse

from src.ho_optim_milp.plotting import plot_pareto_fronts
from src.ho_optim_milp.plotting import utils as ut

DATASET_DIR: str = "dataset_root"


def main() -> int:
    """Load published datasets and plot Pareto frontiers."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim-path",
        default=ut.get_default_optim_result_path(DATASET_DIR),
        help="Path to gp_optim_result_metrics.parquet",
    )
    parser.add_argument(
        "--reference-path",
        default=ut.get_default_reference_result_path(DATASET_DIR),
        help="Path to reference_result_metrics.parquet",
    )
    parser.add_argument(
        "--out-path",
        default=ut.get_default_plot_path(),
        help="Output path for figures.",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print plotted values to stdout.",
    )
    args = parser.parse_args()

    plot_pareto_fronts(
        optim_path=args.optim_path,
        reference_path=args.reference_path,
        out_path=args.out_path,
        print_values=args.print_values,
        annotate_lambdas=True,
    )

    return 0


if __name__ == "__main__":
    main()
