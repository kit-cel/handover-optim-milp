"""Sweep Reference Runner."""

from argparse import _SubParsersAction as SubParsersAct
from datetime import datetime
import os

import wandb

from ho_optim_milp.common.subparsers import add_default_sweep_parser
from ho_optim_milp.reference.rrc_config import RRCConfig
from ho_optim_milp.result_manager.wandb_logger import WandBLogger
from ho_optim_milp.scripts.run_reference import main as run_reference


def add_parser(subparsers: SubParsersAct, name: str = "sweep_reference") -> None:
    """Argument parser for main function."""
    add_default_sweep_parser(
        subparsers, main, name=name, optimization_or_reference="reference"
    )


def wb_run_reference(config: str, dataset: str, **kwargs) -> int:
    """Wrapper for run_reference to be used with wandb agent."""
    cwd = kwargs.get("cwd_path", os.getcwd())
    path_to_config = os.path.join(cwd, "config", config)
    if not os.path.isfile(path_to_config):
        raise ValueError(f"Configuration file '{path_to_config}' does not exist.")
    rrc_config = RRCConfig.from_yaml(path=path_to_config)

    wandb_logger = WandBLogger(
        config=rrc_config,
        run_name=f"run_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        enable=True,
        # project=wb_project,
        # team_name=wb_team,
    )

    ep_idx = wandb.config.ep_idx
    rrc_config.update(
        {
            "ho_event_config": {
                "a3": {
                    "ttt_ms": wandb.config.ttt,
                    "hys": wandb.config.hys,
                    "offset": wandb.config.offset,
                }
            },
        }
    )

    kwargs.update({"rrc_config": rrc_config, "wandb_logger": wandb_logger})

    return run_reference(config="", dataset=dataset, ep_idx=ep_idx, **kwargs)


def main(
    wb_team: str,
    wb_project: str,
    wb_sweep_id: str,
    config: str,
    dataset: str,
    **kwargs,
) -> int:
    """Main function for wandb sweep."""
    print(
        f"W&B Team {wb_team}\n"
        f"W&B Project {wb_project}\n"
        f"W&B Sweep ID: {wb_sweep_id}"
    )

    sweep = f"{wb_team}/{wb_project}/{wb_sweep_id}"
    wandb.agent(
        sweep,
        function=lambda: wb_run_reference(config=config, dataset=dataset, **kwargs),
    )

    return 0
