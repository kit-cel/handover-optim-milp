"""Fast consistency checks for the separately distributed paper dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ho_optim_milp.dataloader.episodes import episode_shapes, load_episode
from ho_optim_milp.optimization.optim_config import OptimConfig


@pytest.mark.paper_dataset
def test_published_network_data_and_metrics_are_consistent() -> None:
    root_env = os.environ.get("PAPER_DATASET_ROOT")
    if not root_env:
        pytest.skip("PAPER_DATASET_ROOT is not set")
    root = Path(root_env)
    network_file = root / "network_data" / "network_simulation_results_060kph_10eps.h5"
    metrics_file = root / "milp_results" / "optim_results_metrics_060kph_10eps.parquet"
    if not network_file.is_file() or not metrics_file.is_file():
        pytest.fail("PAPER_DATASET_ROOT does not contain the published 60 km/h inputs and metrics")
    shapes = episode_shapes(network_file)
    assert len(shapes) == 10
    assert all((n_steps, n_ue, n_bs) == (1500, 100, 21)
               for _, n_steps, n_ue, n_bs in shapes)
    episode = load_episode(network_file, 0)
    assert episode.sinr.shape == (1500, 100, 21)
    assert np.isfinite(episode.sinr[:, 0, :]).all()
    assert np.isfinite(episode.rsrp[:, 0, :]).all()

    published = pd.read_parquet(metrics_file)
    assert len(published) == 4200
    assert not published.duplicated(["episode_idx", "ue_idx", "lambda"]).any()
    assert set(published["episode_idx"]) == set(range(6))
    assert set(published["n311"]) == {10}
    assert set(published["simulated_steps"]) == {1500}
    expected_objective = (
        published["mean_capacity"]
        - published["lambda"] * (1 - published["rel_connected_time"])
    )
    np.testing.assert_allclose(
        published["obj_value"], expected_objective, rtol=0, atol=2e-5
    )

    config_path = Path(__file__).parents[1] / "config" / "optimization_config.yaml"
    config = OptimConfig.from_yaml(str(config_path))
    assert config.rrc.n311 == 10
    assert config.rrc.t_res_ms == int(published["t_res_ms"].iloc[0])
    assert config.rrc.t310_ms == int(published["t310_ms"].iloc[0])
