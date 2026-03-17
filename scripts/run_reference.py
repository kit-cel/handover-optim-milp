"""RRC Simulation"""

import argparse
import csv
import os

from src.ho_optim_milp.common.utils import nested_dict_to_str
from src.ho_optim_milp.reference.reference import RRCReferenceSimulation
from src.ho_optim_milp.reference.rrc_config import RRCConfig
from src.ho_optim_milp.result_manager.simulation_result import SimulationResults


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Argument parser for main function."""
    parser: argparse.ArgumentParser = subparsers.add_parser("run_reference")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="reference_config.yaml",
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
        help="Episode index.",
    )

    parser.set_defaults(func=main)


def main(config: str, dataset: str, ep_idx: int, **kwargs) -> int:
    """Run a single simulation instance."""
    cwd = kwargs.get("cwd", os.getcwd())
    max_steps = kwargs.get("max_steps", 10_000)

    path_to_config = os.path.join(cwd, "config", config)
    if not os.path.isfile(path_to_config):
        raise ValueError(f"Configuration file '{path_to_config}' does not exist.")
    cfg = RRCConfig.from_yaml(path=path_to_config)

    path_to_dataset = os.path.join(cwd, "dataset_root", "network_data", dataset)

    dataset_name = os.path.splitext(os.path.basename(path_to_dataset))[0]
    sim_results = SimulationResults.load(
        path=path_to_dataset,
        max_steps=max_steps,
        keys=["rsrp", "sinr", "ue_pos"],
        validate_data=True,
    )
    ep_result = sim_results.episode_results[ep_idx]

    sim = RRCReferenceSimulation(cfg, ep_result, lambda_r=0, log_full_results=False)
    sim.run()

    # Get results
    agg_result_metrics = sim.get_result_metrics(aggregated=True)
    result_metrics = sim.get_result_metrics(aggregated=False)
    print(f"Results:\n{nested_dict_to_str(agg_result_metrics)}\n")

    meta = {
        "t_res_ms": cfg.t_res_ms,
        "q_in_db": cfg.q_in_db,
        "q_out_db": cfg.q_out_db,
        "n310": cfg.n310,
        "n311": cfg.n311,
        "t310_ms": cfg.t310_ms,
        "t_ho_prep_ms_simulated": cfg.t_ho_prep_ms_simulated,
        "t_ho_exec_ms_simulated": cfg.t_ho_exec_ms_simulated,
        "t_rlfr_ms_simulated": cfg.t_rlfr_ms_simulated,
        "t_mts_ms": cfg.t_mts_ms,
        "l3_filter_coef": cfg.l3_filter_coef,
        "a3_ttt_ms": cfg.ho_event_config["a3"]["ttt_ms"],
        "a3_hys_db": cfg.ho_event_config["a3"]["hys"],
        "a3_off_dbm": cfg.ho_event_config["a3"]["offset"],
    }

    ue_cfg = ep_result.config.get("ue", None)
    if isinstance(ue_cfg, list) and isinstance(ue_cfg[0], dict):
        ue_speed = ue_cfg[0]["speed_kmh"]
    else:
        ue_speed = None

    meta.update(
        {
            "dataset_name": dataset_name,
            "ep_idx": ep_idx,
            "seed": ep_result.config.get("seed", None),
            "ue_speed_kph": ue_speed,
        }
    )
    print(f"Meta:\n{nested_dict_to_str(meta)}\n")

    # Log UE results to CSV including RRC parameters
    log_dir = os.path.join(cfg.base_dir, "results", "reference", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = (
        f"ref_results"
        f"_ttt{meta['a3_ttt_ms']}_hys{meta['a3_hys_db']}_off{meta['a3_off_dbm']}"
        f".csv"
    )
    full_log_path = os.path.join(log_dir, log_file_name)

    # Write UE result metrics to CSV
    exists = os.path.exists(full_log_path)
    with open(full_log_path, "a", newline="", encoding="utf-8") as f:
        header = list(meta.keys()) + list(result_metrics[0].keys())  # type: ignore
        print(header)
        writer = csv.DictWriter(f, fieldnames=header, delimiter=";")

        if not exists:
            writer.writeheader()

        for _, ue_metrics in result_metrics.items():
            row = {}
            row.update(meta)
            row.update(ue_metrics)
            writer.writerow(row)

    return 0
