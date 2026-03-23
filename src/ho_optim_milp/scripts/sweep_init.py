"""Script to initialize a W&B sweep."""

import argparse
from argparse import _SubParsersAction as SubParsersAct
from datetime import datetime

import numpy as np
import wandb


def add_parser(subparsers: SubParsersAct, name: str = "sweep_init") -> None:
    """Argument parser for main function."""
    parser: argparse.ArgumentParser = subparsers.add_parser(name)

    parser.add_argument(
        "-n",
        "--name",
        dest="optimization_or_reference",
        help="Whether to initialize a sweep for the optimization (optim) or "
        "reference (ref) implementation.",
    )

    parser.set_defaults(func=main)


def main(
    optimization_or_reference: str, **kwargs  # pylint: disable=unused-argument
) -> int:
    """Initialize a W&B sweep."""
    if optimization_or_reference not in {"optimization", "reference"}:
        raise ValueError("Invalid argument. Must be 'optimization' or 'reference'.")

    wb_base_config = {
        "name": f"sweep_{optimization_or_reference}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}",
        "method": "grid",
        "metric": {"name": "ho_obj_value", "goal": "maximize"},
        "parameters": {
            "ep_idx": {"values": np.arange(10).tolist()},
            "ue_idx": {"values": np.arange(100).tolist()},
        },
    }
    if optimization_or_reference == "optimization":
        wb_config = wb_base_config.copy()
        wb_config["parameters"].update(
            {
                "lambda_r": {"values": [0.1, 3, 10, 30, 100, 300, 1_000]},
            }
        )
    elif optimization_or_reference == "reference":
        wb_config = wb_base_config.copy()
        wb_config["parameters"].update(
            {
                "ttt": {  # (40 ms steps)
                    "values": [0, 40, 80, 160, 320],
                },
                "hys": {  # 0 to 15 dB (1 dB steps)
                    "values": np.arange(0, 16, 1).tolist(),
                },
                "offset": {  # -15 dB to 15 dB (1 dB steps)
                    "values": np.arange(-15, 16, 1).tolist(),
                },
            }
        )
    else:
        raise ValueError("Invalid argument. Must be 'optimization' or 'reference'.")

    sweep_id = wandb.sweep(sweep=wb_config, project="ho-optim-milp")
    print(f"W&B sweep with ID: {sweep_id}")
    print(f"Sweep name:        {wb_config['name']}")

    return 0
