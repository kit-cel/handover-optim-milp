"""Sweep Optimization Runner."""

from argparse import _SubParsersAction as SubParsersAct
from datetime import datetime
import os

import wandb

from ho_optim_milp.common.subparsers import add_default_sweep_parser
from ho_optim_milp.optimization.optim_config import OptimConfig
from ho_optim_milp.result_manager.wandb_logger import WandBLogger
from ho_optim_milp.scripts.run_optimization import main as run_optimization


def add_parser(subparsers: SubParsersAct, name: str = "sweep_optimization") -> None:
    """Argument parser for main function."""
    add_default_sweep_parser(
        subparsers, main, name=name, optimization_or_reference="optimization"
    )


def wb_run_optimization(config: str, dataset: str, **kwargs) -> int:
    """Wrapper for run_optimization to be used with wandb agent."""
    cwd = kwargs.get("cwd_path", os.getcwd())
    path_to_config = os.path.join(cwd, "config", config)
    if not os.path.isfile(path_to_config):
        raise ValueError(f"Configuration file '{path_to_config}' does not exist.")
    optim_config = OptimConfig.from_yaml(path=path_to_config)

    wandb_logger = WandBLogger(
        config=optim_config,
        run_name=f"run_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        enable=True,
        # project=wb_project,
        # team_name=wb_team,
    )

    ep_idx = wandb.config.ep_idx
    ue_idx = wandb.config.ue_idx
    optim_config.update({"lambda_r": wandb.config.lambda_r})

    kwargs.update({"optim_config": optim_config, "wandb_logger": wandb_logger})

    return run_optimization(
        config="", dataset=dataset, ep_idx=ep_idx, ue_idx=ue_idx, **kwargs
    )


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
        function=lambda: wb_run_optimization(config=config, dataset=dataset, **kwargs),
    )

    return 0
