"""Plot"""

import argparse
import os


from src.plotting import plot_rate_connectivity_tradeoff
from src.plotting import utils as ut

DATASET_ROOT: str = os.path.join(os.path.abspath(__file__), "dataset_root")


def main() -> None:
    """Load published datasets and plot the tradeoff versus lambda."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim-path",
        default=ut.get_default_optim_path(DATASET_ROOT),
        help="Path to gp_optim_result_metrics.parquet",
    )
    parser.add_argument(
        "--reference-path",
        default=ut.get_default_reference_path(DATASET_ROOT),
        help="Path to reference_result_metrics.parquet",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print plotted values to stdout.",
    )
    args = parser.parse_args()

    opt_curve = ut.load_optimization_curve(args.optim_path)
    ref_agg = ut.load_reference_parameter_aggregation(args.reference_path)

    ref_q99 = ut.reference_curve_for_quantile_top_tail(
        ref_agg,
        opt_curve.lam,
        99.0,
    )

    out_path = os.path.join(
        ut.get_default_plots_dir(),
        "rate_connected_time_vs_lambda_q99.png",
    )

    plot_rate_connectivity_tradeoff(
        opt=opt_curve,
        ref_q99=ref_q99,
        out_path=out_path,
        print_values=args.print_values,
    )

    print(out_path)


if __name__ == "__main__":
    main()
