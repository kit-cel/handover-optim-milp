"""CLI for a dataset built entirely from recorded network measurements."""

from argparse import _SubParsersAction as SubParsersAct
from pathlib import Path

from ho_optim_milp.dataset_pipeline import build_dataset
from ho_optim_milp.optimization.optim_config import OptimConfig
from ho_optim_milp.reference.rrc_config import RRCConfig


def add_parser(subparsers: SubParsersAct) -> None:
    parser = subparsers.add_parser("build_dataset")
    parser.add_argument("--network-data", action="append", required=True,
                        metavar="SPLIT=FILE")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--lambda", dest="lambdas", type=float, action="append",
                        required=True)
    parser.add_argument("--a3-config-path", action="append", default=[])
    parser.add_argument("--a3-backend", choices=["standalone", "simulator"],
                        default="standalone")
    parser.add_argument("--output-root", required=True)
    parser.set_defaults(func=main)


def main(
    network_data: list[str], config_path: str, lambdas: list[float],
    a3_config_path: list[str], a3_backend: str, output_root: str, **_: object,
) -> int:
    sources: dict[str, Path] = {}
    for item in network_data:
        split, separator, filename = item.partition("=")
        if not separator or not split or not filename:
            raise ValueError(f"--network-data must be SPLIT=FILE, got {item!r}")
        if split in sources:
            raise ValueError(f"Duplicate split: {split}")
        sources[split] = Path(filename)
    config = OptimConfig.from_yaml(config_path)
    profiles = [RRCConfig.from_yaml(path) for path in a3_config_path] or [config.rrc]
    build_dataset(sources, config, lambdas, output_root,
                  a3_profiles=profiles, a3_backend=a3_backend)
    return 0
