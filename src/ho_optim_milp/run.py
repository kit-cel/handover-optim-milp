"""
Main script to run the handover optimization framework.

This script provides a unified command-line interface to:
- Optimize the MILP on the provided datasets.
- Run the reference on the provided datasets.
- Plot the results.
- Perform sweeps over multiple episodes and UEs.

Usage:
    python -m ho_optim_milp.run <script>

Arguments:
    script: One of .
"""

import argparse
import os
import sys

from ho_optim_milp.scripts import (
    plot_pareto_fronts,
    plot_tradeoff,
    run_optimization,
    run_reference,
    sweep_init,
    sweep_optimization,
    sweep_reference,
)


def main() -> int:
    """Run the Handover Optimization Framework."""
    parser = argparse.ArgumentParser(
        description="Handover Optimization Framework\n\n"
        "Use this entry point to optimize the MILP, run the reference implementation, "
        "and to evaluate the achieved handover performance.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cwd = os.getcwd()

    # register commands
    plot_pareto_fronts.add_parser(subparsers, cwd=cwd)
    plot_tradeoff.add_parser(subparsers, cwd=cwd)

    run_optimization.add_parser(subparsers)
    run_reference.add_parser(subparsers)

    sweep_init.add_parser(subparsers)
    sweep_optimization.add_parser(subparsers)
    sweep_reference.add_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    kwargs = vars(args).copy()
    func = kwargs.pop("func")
    kwargs.pop("command", None)

    kwargs["cwd_path"] = cwd

    return func(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
