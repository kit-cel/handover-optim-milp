"""Common subparser definitions for scripts."""

import argparse
from argparse import _SubParsersAction as SubParsersAct
from typing import Callable

from ho_optim_milp.plotting import utils as ut

DATASET_DIR: str = "dataset_root"


def add_default_plot_parser(
    subparsers: SubParsersAct,
    main: Callable[..., int],
    cwd: str,
    name: str,
) -> None:
    """Argument parser for main function."""
    parser: argparse.ArgumentParser = subparsers.add_parser(name)

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


def add_default_simulation_parser(
    subparsers: SubParsersAct,
    main: Callable[..., int],
    name: str,
    optimization_or_reference: str = "optimization",
) -> None:
    """Argument parser for main function."""
    if optimization_or_reference not in ["optimization", "reference"]:
        raise ValueError("Invalid value for optimization_or_reference.")

    parser: argparse.ArgumentParser = subparsers.add_parser(name)

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=f"{optimization_or_reference}_config.yaml",
        help="Name of the configuration file.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="network_results.h5",
        dest="dataset",
        help="Name of the Dataset.",
    )
    parser.add_argument(
        "--ep-idx",
        dest="ep_idx",
        type=int,
        required=True,
        help="Episode index.",
    )
    if optimization_or_reference == "optimization":
        parser.add_argument(
            "--ue-idx",
            dest="ue_idx",
            type=int,
            required=True,
            help="Index of the UE in the dataset.",
        )

    parser.set_defaults(func=main)


def add_default_sweep_parser(
    subparsers: SubParsersAct,
    main: Callable[..., int],
    name: str,
    optimization_or_reference: str = "optimization",
) -> None:
    """Argument parser for main function."""
    if optimization_or_reference not in ["optimization", "reference"]:
        raise ValueError("Invalid value for optimization_or_reference.")

    parser: argparse.ArgumentParser = subparsers.add_parser(name)

    parser.add_argument(
        "--team",
        dest="wb_team",
        type=str,
        required=True,
        help="Weights & Biases team name.",
    )
    parser.add_argument(
        "--project",
        dest="wb_project",
        type=str,
        required=True,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--sweep-id",
        dest="wb_sweep_id",
        type=str,
        required=True,
        help="Weights & Biases sweep ID.",
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=f"{optimization_or_reference}_config.yaml",
        help="Name of the configuration file.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="network_results.h5",
        dest="dataset",
        help="Name of the Dataset.",
    )

    parser.set_defaults(func=main)
