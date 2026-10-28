"""CLI for optional simulator-backed network trace generation."""

from argparse import _SubParsersAction as SubParsersAct

from ho_optim_milp.integrations.simulator import generate_network_data


def add_parser(subparsers: SubParsersAct) -> None:
    parser = subparsers.add_parser("generate_network_data")
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--output-path", required=True)
    parser.set_defaults(func=main)


def main(sim_config: str, output_path: str, **_: object) -> int:
    generate_network_data(sim_config, output_path)
    return 0
