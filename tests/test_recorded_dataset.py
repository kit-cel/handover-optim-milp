"""A short recorded trajectory produces a complete schema-3 release."""

import json
from pathlib import Path

import gurobipy as gp
import h5py
import numpy as np
import pandas as pd
import pytest

from ho_optim_milp.dataset_generator.recorded import build_recorded_dataset
from ho_optim_milp.dataset import Dataset, validate_dataset
from ho_optim_milp.cli import main as cli_main

from test_model_profiles import make_config


def _short_network_data(path: Path) -> None:
    sinr = np.asarray([[12, 0]] * 4 + [[0, 12]] * 16, dtype=np.float32)[:, None, :]
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("config_common", data=json.dumps({"t_res_ms": 40}))
        meta.create_dataset("release", data=json.dumps({"schema_version": 3}))
        episode = handle.create_group("ep_0000")
        for name, values in {
            "sinr": sinr, "rsrp": sinr - 70.0,
            "ue_pos": np.zeros((20, 1, 2), dtype=np.float32),
            "distances_2d": np.zeros_like(sinr),
            "distances_3d": np.zeros_like(sinr),
            "ue_move_angle": np.zeros((20, 1), dtype=np.float32),
        }.items():
            episode.create_dataset(name, data=values)
        episode.create_group("meta").create_dataset(
            "config_full", data=json.dumps({
                "t_res_ms": 40,
                "ues": [{"gauss_markov_mean_speed_kph": 60.0}],
            }),
        )


def test_short_recorded_data_builds_release_without_simulator(tmp_path: Path) -> None:
    try:
        model = gp.Model("license_probe")
        model.dispose()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi licence unavailable: {error}")
    source = tmp_path / "network.h5"
    _short_network_data(source)
    config = make_config()
    root = build_recorded_dataset(
        {"id_val": source}, config, [1.0], tmp_path / "sample_release",
        a3_profiles=[config.rrc], selection_split="id_val",
        speed_bins=((0.0, 200.0),),
    )
    assert validate_dataset(root, verify_artifact_hashes=True)["artifact_count"] > 0
    assert pd.read_parquet(root / "optim_results/id_val/optim_results.parquet").shape[0] == 20
    split = Dataset.open(root).split("id_val")
    binary = split.milp_metrics("binary_rank1")
    assert len(binary) == 1
    assert len(split.milp_trajectory(
        episode_idx=0, ue_id=0, lambda_value=1.0, variant="binary_rank1",
    )) == 20
    assert set(pd.read_parquet(root / "reference_results/a3/selections.parquet")["level"]) == {
        "global", "speed_aware", "split_oracle"
    }
    assert (root / "events/id_val/handover_attempts.parquet").is_file()
    assert (root / "plots/id_val/event_rates_vs_lambda.png").is_file()
    output = tmp_path / "separate_plots"
    assert cli_main([
        "plot-dataset", "--dataset-root", str(root), "--split", "id_val",
        "--output-root", str(output),
    ]) == 0
    assert (output / "plots/id_val/event_rates_vs_lambda.png").is_file()
    assert validate_dataset(root)["artifact_count"] > 0
