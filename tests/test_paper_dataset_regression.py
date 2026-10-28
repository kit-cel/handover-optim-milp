"""Optional comparison with the separately distributed paper dataset."""

import os
from pathlib import Path

import gurobipy as gp
import pandas as pd
import pytest

from ho_optim_milp.dataloader.episodes import load_episode
from ho_optim_milp.optimization.milp import HandoverOptimizerMILP
from ho_optim_milp.optimization.optim_config import OptimConfig


@pytest.mark.paper_dataset
def test_published_paper_objective_is_reproduced() -> None:
    root_env = os.environ.get("PAPER_DATASET_ROOT")
    if not root_env:
        pytest.skip("PAPER_DATASET_ROOT is not set")
    root = Path(root_env)
    network_file = root / "network_data" / "network_simulation_results_060kph_10eps.h5"
    metrics_file = root / "milp_results" / "optim_results_metrics_060kph_10eps.parquet"
    if not network_file.is_file() or not metrics_file.is_file():
        pytest.fail("PAPER_DATASET_ROOT does not contain the published 60 km/h inputs and metrics")
    try:
        probe = gp.Model("license_probe")
        probe.dispose()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi licence unavailable: {error}")

    published = pd.read_parquet(metrics_file)
    match = published.loc[(published["episode_idx"] == 0)
                          & (published["ue_idx"] == 0)
                          & (published["lambda"] == 0.1)]
    assert len(match) == 1
    row = match.iloc[0]
    assert int(row["n311"]) == 10

    config_path = Path(__file__).parents[1] / "config" / "optimization_config.yaml"
    config = OptimConfig.from_yaml(str(config_path))
    assert config.rrc.n311 == 10
    config.lambda_r = 0.1
    config.gurobi.threads = 2
    config.gurobi.log_to_console = False
    optimizer = HandoverOptimizerMILP(config)
    optimizer.load_data(load_episode(network_file, 0), ue_idx=0)
    optimizer.setup_model()
    optimizer.solve()
    assert optimizer.model.Status == gp.GRB.OPTIMAL
    assert optimizer.model.ObjVal == pytest.approx(float(row["obj_value"]), abs=0.01)
