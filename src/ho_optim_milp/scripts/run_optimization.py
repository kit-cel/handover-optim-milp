"""Gurobi Optimization Runner."""

import argparse
import csv
import os

from ho_optim_milp.common.utils import nested_dict_to_str
from ho_optim_milp.optimization.optim_config import OptimConfig
from ho_optim_milp.optimization.milp import HandoverOptimizerMILP
from ho_optim_milp.result_manager.simulation_result import SimulationResults


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Argument parser for main function."""
    parser: argparse.ArgumentParser = subparsers.add_parser("run_optimization")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="optimization_config.yaml",
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
    parser.add_argument(
        "--ue-idx",
        dest="ue_idx",
        type=int,
        required=True,
        help="Index of the UE in the dataset.",
    )

    parser.set_defaults(func=main)


def main(config: str, dataset: str, ep_idx: int, ue_idx: int, **kwargs) -> int:
    """Run a single simulation instance."""
    cwd = kwargs.get("cwd", os.getcwd())
    max_steps = kwargs.get("max_steps", 10_000)

    path_to_config = os.path.join(cwd, "config", config)
    if not os.path.isfile(path_to_config):
        raise ValueError(f"Configuration file '{path_to_config}' does not exist.")
    cfg = OptimConfig.from_yaml(path=path_to_config)

    path_to_dataset = os.path.join(cwd, "dataset_root", "network_data", dataset)

    sim_results = SimulationResults.load(
        path=path_to_dataset,
        max_steps=max_steps,
        keys=["rsrp", "sinr", "ue_pos"],
        validate_data=True,
    )
    ep_result = sim_results.episode_results[ep_idx]

    milp_solver = HandoverOptimizerMILP(
        config=cfg,
        data=ep_result,
        ue_idx=ue_idx,
    )
    milp_solver.optimize()

    # if not optimal, return early without logging results (2 is optimal)
    if milp_solver.model.status != 2:
        print(
            f"Optimization for episode {ep_idx}, "
            f"UE {ue_idx} did not find an optimal solution. "
            f"Status: {milp_solver.model.status}"
        )
        return 1

    # Get results
    result_metrics = milp_solver.get_result_metrics()
    print(f"Results:\n{nested_dict_to_str(result_metrics)}\n")

    # Log raw optimization results to CSV
    full_log_file_name = (
        f"optim_full_result_ue{ue_idx}_lambda{milp_solver.lambda_r}.csv"
    )
    milp_solver.log_results_to_csv(
        file_name=full_log_file_name, subfolder=f"ep_{ep_idx:05d}"
    )

    meta = {
        "t_res_ms": cfg.rrc.t_res_ms,
        "q_in_db": cfg.rrc.q_in_db,
        "q_out_db": cfg.rrc.q_out_db,
        "n310": cfg.rrc.n310,
        "n311": cfg.rrc.n311,
        "t310_ms": cfg.rrc.t310_ms,
        "t_ho_prep_ms_simulated": cfg.rrc.t_ho_prep_ms_simulated,
        "t_ho_exec_ms_simulated": cfg.rrc.t_ho_exec_ms_simulated,
        "t_rlfr_ms_simulated": cfg.rrc.t_rlfr_ms_simulated,
        "t_mts_ms": cfg.rrc.t_mts_ms,
        "l3_filter_coef": cfg.rrc.l3_filter_coef,
        "lambda": milp_solver.lambda_r,
    }

    ue_cfg = ep_result.config.get("ue", None)
    if isinstance(ue_cfg, list) and isinstance(ue_cfg[0], dict):
        ue_speed = ue_cfg[0]["speed_kmh"]
    else:
        ue_speed = None

    meta.update(
        {
            "dataset_name": os.path.splitext(os.path.basename(path_to_dataset))[0],
            "ep_idx": ep_idx,
            "ue_idx": ue_idx,
            "seed": ep_result.config.get("seed", None),
            "ue_speed_kph": ue_speed,
        }
    )
    print(f"Meta:\n{nested_dict_to_str(meta)}\n")

    # Log UE results metrics to CSV including RRC parameters
    log_dir = os.path.join(cfg.base_dir, "results", "optimization")
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"metrics_ep{ep_idx}_lambda{milp_solver.lambda_r}.csv"
    full_log_path = os.path.join(log_dir, log_file_name)

    # Write UE result metrics to CSV
    file_exists = os.path.exists(full_log_path)
    with open(full_log_path, "a", newline="", encoding="utf-8") as f:
        header = list(meta.keys()) + list(result_metrics.keys())
        writer = csv.DictWriter(f, fieldnames=header, delimiter=";")

        if not file_exists:
            writer.writeheader()

        row = {}
        row.update(meta)
        row.update(result_metrics)
        writer.writerow(row)

    return 0
