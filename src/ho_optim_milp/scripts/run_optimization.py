"""Gurobi Optimization Runner."""

from argparse import _SubParsersAction as SubParsersAct
import os
import warnings

from ho_optim_milp.common.subparsers import add_default_simulation_parser
from ho_optim_milp.common.utils import nested_dict_to_str
from ho_optim_milp.optimization.optim_config import OptimConfig
from ho_optim_milp.optimization.milp import HandoverOptimizerMILP
from ho_optim_milp.result_manager.result_metrics import ResultsMetrics
from ho_optim_milp.result_manager.network_simulation_result import (
    NetworkSimulationResult,
)
from ho_optim_milp.result_manager.wandb_logger import WandBLogger


def add_parser(subparsers: SubParsersAct, name: str = "run_optimization") -> None:
    """Argument parser for main function."""
    add_default_simulation_parser(
        subparsers, main, name=name, optimization_or_reference="optimization"
    )


def main(config: str, dataset: str, ep_idx: int, ue_idx: int, **kwargs) -> int:
    """Run a single simulation instance."""
    cwd = kwargs.get("cwd_path", os.getcwd())
    max_steps = kwargs.get("max_steps", 10_000)
    wandb_logger: WandBLogger | None = kwargs.get("wandb_logger", None)
    optim_config: OptimConfig | None = kwargs.get("optim_config", None)

    path_to_config = os.path.join(cwd, "config", config)
    if isinstance(optim_config, OptimConfig):
        if config != "":
            warnings.warn(
                f"Warning: Configuration file '{path_to_config}' is ignored since "
                "optim_config is provided directly. Use an empty string for the "
                "config argument to avoid this warning."
            )
        print(f"Use configuration:\n{nested_dict_to_str(optim_config.to_dict())}\n")
    else:
        if not os.path.isfile(path_to_config):
            raise ValueError(f"Configuration file '{path_to_config}' does not exist.")
        optim_config = OptimConfig.from_yaml(path=path_to_config)

    sim_results = NetworkSimulationResult.load(
        path=os.path.join(cwd, "dataset_root", "network_data", dataset),
        max_steps=max_steps,
        keys=["rsrp", "sinr", "ue_pos"],
        validate_data=True,
    )
    ep_result = sim_results.get_ep_result_by_idx(ep_idx)

    milp_solver = HandoverOptimizerMILP(config=optim_config)
    milp_solver.load_data(ep_result=ep_result, ue_idx=ue_idx)
    milp_solver.setup_model(name=f"ho_optim_ep{ep_idx}_ue{ue_idx}")
    milp_solver.apply_warm_start_if_configured()
    milp_solver.solve()

    # if not optimal, return early without logging results (2 is optimal)
    if milp_solver.model.status != 2:
        print(
            f"Optimization for episode {ep_idx}, "
            f"UE {ue_idx} did not find an optimal solution. "
            f"Status: {milp_solver.model.status}"
        )
        return 1

    # Log full optimization results to CSV
    milp_solver.log_results_to_csv(
        file_name=f"optim_full_result_ue{ue_idx}_lambda{optim_config.lambda_r}.csv",
        subfolder=f"ep_{ep_idx:04d}",
    )

    # Get result metrics and log to CSV with metadata
    agg_result_metrics = milp_solver.get_aggregated_result_metrics()
    print(f"Result metrics:\n{nested_dict_to_str(agg_result_metrics)}\n")

    result_metrics = ResultsMetrics(config=optim_config, ep_idx=ep_idx)
    result_metrics.add_result_metrics(milp_solver.get_result_metrics())
    result_metrics.add_aggregated_result_metrics(agg_result_metrics)
    result_metrics.add_meta(
        {
            "t_res_ms": optim_config.rrc.t_res_ms,
            "q_in_db": optim_config.rrc.q_in_db,
            "q_out_db": optim_config.rrc.q_out_db,
            "n310": optim_config.rrc.n310,
            "n311": optim_config.rrc.n311,
            "t310_ms": optim_config.rrc.t310_ms,
            "t_ho_prep_ms_simulated": optim_config.rrc.t_ho_prep_ms_simulated,
            "t_ho_exec_ms_simulated": optim_config.rrc.t_ho_exec_ms_simulated,
            "t_rlfr_ms_simulated": optim_config.rrc.t_rlfr_ms_simulated,
            "t_mts_ms": optim_config.rrc.t_mts_ms,
            "l3_filter_coef": optim_config.rrc.l3_filter_coef,
            "lambda": optim_config.lambda_r,
            "dataset": dataset,
            "ep_name": ep_result.ep_name,
            "seed": ep_result.config.get("seed", None),
            "ue_speed_kph": ep_result.extract_ue_speed_kph(),
        }
    )
    if isinstance(wandb_logger, WandBLogger) and wandb_logger.run is not None:
        result_metrics.add_meta(
            {
                "wandb_run_id": f"{wandb_logger.run.id}",
                "wandb_sweep_id": f"{wandb_logger.run.sweep_id}",
            },
        )

    result_metrics.save_metrics_to_csv(
        subfolder="optimization",
        filename_include_meta=["wandb_sweep_id", "lambda"],
    )

    # Log results to WandB if wandb_logger is provided
    if isinstance(wandb_logger, WandBLogger) and wandb_logger.run is not None:
        wandb_logger.log(result_metrics.get_agg_metrics_with_meta(), step=int(ue_idx))
        wandb_logger.finish_run()

    return 0
