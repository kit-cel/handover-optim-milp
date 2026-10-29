"""Command-line interface for optimization on recorded network data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd
import yaml
import mobile_network_sim
from mobile_network_sim.rrc import RRCConfig

from . import __version__
from .dataloader.episodes import load_episode
from .dataset import Dataset, validate_dataset
from .dataset_generator.a3_protocol import DEFAULT_A3_PROTOCOL
from .dataset_generator.recorded import build_recorded_dataset
from .dataset_generator.recorded_results import (
    optimize_recorded_ue,
    reference_recorded_episode,
)
from .dataset_generator.reporting import publish_reports
from .dataset_generator.storage import validate_episode, write_json, write_parquet
from .integrations.simulator import generate_network_data
from .optimization.optim_config import OptimConfig


def _add_recorded_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the public ``ho-optim`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="ho-optim",
        description="Optimize and evaluate recorded mobile-network trajectories.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    optimize = commands.add_parser("optimize", help="Optimize one recorded UE.")
    _add_recorded_input(optimize)
    optimize.add_argument("--ue-index", type=int, required=True)

    reference = commands.add_parser("run-a3", help="Replay A3 on one recorded episode.")
    _add_recorded_input(reference)

    build = commands.add_parser(
        "build-dataset", help="Build a complete dataset from recorded HDF5 files."
    )
    build.add_argument("--network-data", action="append", required=True,
                       metavar="SPLIT=FILE")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--lambda", dest="lambdas", type=float, action="append",
                       required=True)
    build.add_argument("--a3-config", type=Path, action="append", required=True)
    build.add_argument("--a3-search", action="store_true",
                       help="Use the recorded coarse-to-fine A3 search protocol.")
    build.add_argument("--selection-split", required=True)
    build.add_argument("--output-root", type=Path, required=True)

    validate = commands.add_parser("validate-dataset", help="Validate a dataset release.")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--verify-artifact-hashes", action="store_true")

    plot = commands.add_parser(
        "plot-dataset", help="Create result figures from a completed dataset."
    )
    plot.add_argument("--dataset-root", type=Path, required=True)
    plot.add_argument("--split", required=True)
    plot.add_argument("--output-root", type=Path, required=True)

    generate = commands.add_parser(
        "generate-network-data", help="Generate radio data with the simulator."
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    return parser


def _recorded_episode(args: argparse.Namespace, rrc_config: RRCConfig):
    if args.episode_index < 0:
        raise ValueError("episode-index must be non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    episode = load_episode(args.network_data, args.episode_index,
                           max_steps=args.max_steps)
    validate_episode(episode, rrc_config)
    return episode


def _output_directory(path: Path) -> Path:
    output = path.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _command_optimize(args: argparse.Namespace) -> int:
    config = OptimConfig.from_yaml(str(args.config))
    episode = _recorded_episode(args, config.rrc)
    if args.ue_index < 0:
        raise ValueError("ue-index must be non-negative")
    metrics, steps = optimize_recorded_ue(episode, args.ue_index, config)
    output = _output_directory(args.output_dir)
    write_parquet(output / "metrics.parquet", pd.DataFrame([metrics]))
    write_parquet(output / "trajectory.parquet", steps)
    write_json(output / "run.json", {
        "method": "milp", "episode_index": args.episode_index,
        "ue_index": args.ue_index, "network_data": args.network_data.name,
        "config": args.config.name, "max_steps": args.max_steps,
        "software": {"ho_optim_milp": __version__,
                     "mobile_network_sim": mobile_network_sim.__version__},
    })
    print(f"Saved MILP result to {output}")
    return 0


def _command_run_a3(args: argparse.Namespace) -> int:
    config = RRCConfig.from_yaml(str(args.config))
    episode = _recorded_episode(args, config)
    results = reference_recorded_episode(
        episode, config, lambda_r=0.0
    )
    output = _output_directory(args.output_dir)
    metrics = pd.DataFrame([row for row, _steps in results])
    steps = pd.concat(
        [frame.assign(ue_id=index) for index, (_row, frame) in enumerate(results)],
        ignore_index=True,
    )
    write_parquet(output / "metrics.parquet", metrics)
    write_parquet(output / "trajectories.parquet", steps)
    write_json(output / "run.json", {
        "method": "a3", "rrc_engine": "mobile-network-sim",
        "episode_index": args.episode_index,
        "network_data": args.network_data.name,
        "config": args.config.name, "max_steps": args.max_steps,
        "software": {"ho_optim_milp": __version__,
                     "mobile_network_sim": mobile_network_sim.__version__},
    })
    print(f"Saved A3 results to {output}")
    return 0


def _command_build_dataset(args: argparse.Namespace) -> int:
    sources: dict[str, Path] = {}
    for item in args.network_data:
        split, separator, filename = item.partition("=")
        if not separator or not split or not filename:
            raise ValueError(f"network-data must be SPLIT=FILE: {item!r}")
        if split in sources:
            raise ValueError(f"Duplicate split: {split}")
        sources[split] = Path(filename)
    config = OptimConfig.from_yaml(str(args.config))
    profiles = [RRCConfig.from_yaml(str(path)) for path in args.a3_config]
    output = build_recorded_dataset(
        sources, config, args.lambdas, args.output_root,
        a3_profiles=profiles, selection_split=args.selection_split,
        a3_protocol=DEFAULT_A3_PROTOCOL if args.a3_search else None,
    )
    print(f"Built dataset at {output}")
    return 0


def _command_validate_dataset(args: argparse.Namespace) -> int:
    result = validate_dataset(
        args.dataset_root, verify_artifact_hashes=args.verify_artifact_hashes
    )
    print(f"Validated {result['artifact_count']} artifacts for {result['release_id']}")
    return 0


def _command_plot_dataset(args: argparse.Namespace) -> int:
    dataset = Dataset.open(args.dataset_root)
    split = dataset.split(args.split)
    output = args.output_root.expanduser().resolve()
    if output == dataset.root or dataset.root in output.parents:
        raise ValueError("Plot output must be outside the validated dataset root")
    if output.exists():
        raise FileExistsError(output)

    rrc = dataset.metadata.get("optim_config", {}).get("rrc", {})
    if not rrc:
        config_path = dataset.artifact_path("simulator_config", split=args.split)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rrc = config.get("rrc", {}) if isinstance(config, dict) else {}
    if not {"q_in_db", "q_out_db"} <= set(rrc):
        raise ValueError("Dataset metadata does not declare Q-in and Q-out thresholds")

    metrics = pd.concat(
        [split.milp_metrics(), split.no_handover_metrics(),
         *(split.a3_metrics(level) for level in
           ("global", "speed_aware", "split_oracle"))],
        ignore_index=True,
    )
    attempts = split.events("handover_attempts")
    outage = split.events("outage_decomposition")
    output.mkdir(parents=True)
    publish_reports(
        metrics=metrics, attempts=attempts, outage=outage,
        output_root=output, q_in_db=float(rrc["q_in_db"]),
        q_out_db=float(rrc["q_out_db"]), report_splits=(args.split,),
    )
    print(f"Saved result plots to {output / 'plots' / args.split}")
    return 0


def _command_generate_network_data(args: argparse.Namespace) -> int:
    output = generate_network_data(args.config, args.output)
    print(f"Saved network data to {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public command-line interface."""
    args = build_parser().parse_args(argv)
    handlers = {
        "optimize": _command_optimize,
        "run-a3": _command_run_a3,
        "build-dataset": _command_build_dataset,
        "validate-dataset": _command_validate_dataset,
        "plot-dataset": _command_plot_dataset,
        "generate-network-data": _command_generate_network_data,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
