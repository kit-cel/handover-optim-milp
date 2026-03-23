"""RRC Simulation"""

from argparse import _SubParsersAction as SubParsersAct
import os
import warnings

from ho_optim_milp.common.subparsers import add_default_simulation_parser
from ho_optim_milp.common.utils import nested_dict_to_str
from ho_optim_milp.reference.reference import RRCReferenceSimulation
from ho_optim_milp.reference.rrc_config import RRCConfig
from ho_optim_milp.result_manager.result_metrics import ResultsMetrics
from ho_optim_milp.result_manager.network_simulation_result import (
    NetworkSimulationResult,
)
from ho_optim_milp.result_manager.wandb_logger import WandBLogger


def add_parser(subparsers: SubParsersAct, name: str = "run_reference") -> None:
    """Argument parser for main function."""
    add_default_simulation_parser(
        subparsers, main, name=name, optimization_or_reference="reference"
    )


def main(config: str, dataset: str, ep_idx: int, **kwargs) -> int:
    """Run a single simulation instance."""
    cwd = kwargs.get("cwd_path", os.getcwd())
    max_steps = kwargs.get("max_steps", 10_000)
    wandb_logger: WandBLogger | None = kwargs.get("wandb_logger", None)
    rrc_config: RRCConfig | None = kwargs.get("rrc_config", None)

    path_to_config = os.path.join(cwd, "config", config)
    if isinstance(rrc_config, RRCConfig):
        if config != "":
            warnings.warn(
                f"Warning: Configuration file '{path_to_config}' is ignored since "
                "rrc_config is provided directly. Use an empty string for the "
                "config argument to avoid this warning."
            )
        print(f"Use configuration:\n{nested_dict_to_str(rrc_config.to_dict())}\n")
    else:
        if not os.path.isfile(path_to_config):
            raise ValueError(f"Configuration file '{path_to_config}' does not exist.")
        rrc_config = RRCConfig.from_yaml(path=path_to_config)

    sim_results = NetworkSimulationResult.load(
        path=os.path.join(cwd, "dataset_root", "network_data", dataset),
        max_steps=max_steps,
        keys=["rsrp", "sinr", "ue_pos"],
        validate_data=True,
    )
    ep_result = sim_results.get_ep_result_by_idx(ep_idx)

    sim = RRCReferenceSimulation(
        rrc_config, ep_result, lambda_r=0, log_full_results=False
    )
    sim.run()

    # Get results
    agg_result_metrics = sim.get_aggregated_result_metrics()
    print(f"Result metrics:\n{nested_dict_to_str(agg_result_metrics)}\n")

    # Get result metrics and log to CSV with metadata
    result_metrics = ResultsMetrics(config=rrc_config, ep_idx=ep_idx)
    result_metrics.add_result_metrics(sim.get_per_ue_result_metrics())
    result_metrics.add_aggregated_result_metrics(agg_result_metrics)
    result_metrics.add_meta(
        {
            "t_res_ms": rrc_config.t_res_ms,
            "q_in_db": rrc_config.q_in_db,
            "q_out_db": rrc_config.q_out_db,
            "n310": rrc_config.n310,
            "n311": rrc_config.n311,
            "t310_ms": rrc_config.t310_ms,
            "t_ho_prep_ms_simulated": rrc_config.t_ho_prep_ms_simulated,
            "t_ho_exec_ms_simulated": rrc_config.t_ho_exec_ms_simulated,
            "t_rlfr_ms_simulated": rrc_config.t_rlfr_ms_simulated,
            "t_mts_ms": rrc_config.t_mts_ms,
            "l3_filter_coef": rrc_config.l3_filter_coef,
            "a3_ttt_ms": rrc_config.ho_event_config["a3"]["ttt_ms"],
            "a3_hys_db": rrc_config.ho_event_config["a3"]["hys"],
            "a3_off_dbm": rrc_config.ho_event_config["a3"]["offset"],
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
        subfolder="reference",
        filename_include_meta=["wandb_sweep_id"],
        # filename_include_meta=["a3_ttt_ms", "a3_hys_db", "a3_off_dbm"],
    )

    # Log results to WandB if wandb_logger is provided
    if isinstance(wandb_logger, WandBLogger) and wandb_logger.run is not None:
        wandb_logger.log(result_metrics.get_agg_metrics_with_meta())
        wandb_logger.finish_run()

    return 0
