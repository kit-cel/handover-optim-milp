"""Plot average achieved rate-outage Pareto frontiers."""

import argparse
import os

import numpy as np


from src.plotting import plot_pareto_fronts
from src.plotting import utils as ut

REF_SCATTER_PERCENTILES: list[float] = [90.0, 95.0, 99.0]

DATASET_ROOT: str = os.path.join(os.path.abspath(__file__), "dataset_root")


def _get_sorted_clipped_percentiles(pcts: list[float] | None) -> list[float]:
    """Clip percentiles to [0, 100] and sort uniquely."""
    if pcts is None:
        return []
    return sorted({float(np.clip(float(p), 0.0, 100.0)) for p in pcts})


def main() -> None:
    """Load published datasets and plot Pareto frontiers."""
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

    ref_percentiles = _get_sorted_clipped_percentiles(REF_SCATTER_PERCENTILES)

    opt_curve = ut.load_optimization_curve(args.optim_path)
    ref_agg = ut.load_reference_parameter_aggregation(args.reference_path)
    bound_conn, bound_cap = ut.reference_pareto_bound(ref_agg)

    ref_curves: list[ut.Curve] = []
    for q in ref_percentiles:
        curve = ut.reference_curve_for_quantile_top_tail(
            ref_agg,
            opt_curve.lam,
            q,
        )
        ref_curves.append(curve)

    pct_tag = "q-" + "-".join(f"{q:g}" for q in ref_percentiles)
    out_path = os.path.join(
        ut.get_default_plots_dir(),
        f"rate_outage_pareto_frontiers_{pct_tag}.png",
    )

    plot_pareto_fronts(
        opt=opt_curve,
        ref_curves=ref_curves,
        bound_conn=bound_conn,
        bound_cap=bound_cap,
        out_path=out_path,
        print_values=args.print_values,
        annotate_lambdas=True,
    )

    print(out_path)


if __name__ == "__main__":
    main()
