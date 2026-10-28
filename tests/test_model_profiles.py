"""Small solver checks for the paper and current MILP profiles."""

import numpy as np
import pytest
import gurobipy as gp

from ho_optim_milp.optimization.milp import HandoverOptimizerMILP
from ho_optim_milp.optimization.milp_result_extractor import (
    _binary_solution,
    _integer_solution,
)
from ho_optim_milp.optimization.optim_config import OptimConfig
from ho_optim_milp.reference.rrc_config import RRCConfig
from ho_optim_milp.result_manager.episode_result import EpisodeResult


@pytest.fixture
def gurobi_license() -> None:
    """Keep model tests distinguishable from an unavailable solver licence."""
    try:
        model = gp.Model("license_probe")
        model.dispose()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi licence unavailable: {error}")


def make_config(profile: str = "paper_v1", **options: object) -> OptimConfig:
    rrc = RRCConfig(
        t_res_ms=40,
        n310=1,
        n311=10,
        t310_ms=120,
        t_ho_prep_ms_simulated=40,
        t_ho_exec_ms_simulated=40,
        t_rlfr_ms_simulated=80,
        t_mts_ms=120,
        l1_q_in_filter_len_ms=80,
        l1_q_out_filter_len_ms=80,
        ho_event_config={"a3": {"ttt_ms": 40, "hys": 0.0, "offset": 1.0}},
    )
    return OptimConfig(
        lambda_r=1.0,
        model_profile=profile,
        rrc=rrc,
        gurobi={"threads": 1, "log_to_console": False},
        **options,
    )


def make_episode() -> EpisodeResult:
    sinr = np.array(
        [[10.0, 0.0], [10.0, 0.0], [8.0, 3.0], [5.0, 8.0],
         [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        dtype=np.float32,
    )[:, None, :]
    n_steps = sinr.shape[0]
    return EpisodeResult(
        ep_name="ep_0000",
        ue_pos=np.zeros((n_steps, 1, 2), dtype=np.float32),
        distances_2d=np.zeros_like(sinr),
        distances_3d=np.zeros_like(sinr),
        rsrp=(sinr - 70.0).astype(np.float32),
        sinr=sinr,
        ue_move_angle=np.zeros((n_steps, 1), dtype=np.float32),
    )


def test_solver_integer_export_rounds_numerical_noise() -> None:
    np.testing.assert_array_equal(_binary_solution([0.00001, 0.99999]), [0, 1])
    np.testing.assert_array_equal(_integer_solution([1.99999, 3.00001]), [2, 3])


def test_new_options_require_current_profile() -> None:
    with pytest.raises(ValueError, match="require model_profile"):
        make_config(initial_access="strongest_rsrp")


@pytest.mark.parametrize("option", [
    {"discount_gamma": 0.9},
    {"initial_access": "strongest_rsrp"},
    {"handover_target_mode": "rank1_neighbor"},
])
def test_each_new_model_option_is_explicit(option: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="require model_profile"):
        make_config(**option)
    config = make_config("current", **option)
    assert config.model_profile == "current"


def test_paper_and_current_preprocessing_differ_without_solver() -> None:
    episode = make_episode()
    paper = HandoverOptimizerMILP(make_config())
    current = HandoverOptimizerMILP(make_config("current"))
    paper.load_data(episode, ue_idx=0)
    current.load_data(episode, ue_idx=0)
    assert paper.rrc_consts.n311 == 10
    np.testing.assert_array_equal(current.data.sinr_db, episode.sinr[:, 0, :].T)
    assert not np.array_equal(paper.data.sinr_db, current.data.sinr_db)
    assert paper.config.use_l3_filtered_sinr is True
    assert current.config.use_l3_filtered_sinr is False


def test_explicit_signal_source_cannot_disagree_with_profile() -> None:
    with pytest.raises(ValueError, match="conflicts with model_profile"):
        make_config("current", use_l3_filtered_sinr=True)


@pytest.mark.parametrize("profile", ["paper_v1", "current"])
def test_profile_solves_and_exports_consistent_metrics(
    profile: str, gurobi_license: None
) -> None:
    solver = HandoverOptimizerMILP(make_config(profile))
    solver.load_data(make_episode(), ue_idx=0)
    solver.setup_model()
    solver.solve()
    assert solver.model.SolCount > 0
    arrays = solver.extract_results()
    metrics = solver.get_aggregated_result_metrics()
    assert arrays["x_bs"].shape == (2, 8)
    np.testing.assert_array_equal(arrays["x_bs"].sum(axis=0), 1)
    assert metrics["num_ho"] == int(arrays["i_ho_exec_end"].sum())
    assert metrics["num_rlf"] == int(arrays["i_rlf_start"].sum())
    assert metrics["pp_rate"] >= 0 or np.isnan(metrics["pp_rate"])


def test_current_profile_uses_raw_sinr_and_rank1_initial_access(
    gurobi_license: None,
) -> None:
    config = make_config(
        "current", initial_access="strongest_rsrp", handover_target_mode="rank1_neighbor"
    )
    solver = HandoverOptimizerMILP(config)
    episode = make_episode()
    solver.load_data(episode, ue_idx=0)
    np.testing.assert_array_equal(solver.data.sinr_db, episode.sinr[:, 0, :].T)
    solver.setup_model()
    solver.solve()
    assert solver.model.SolCount > 0
    assert solver.extract_results()["pcell"][0] == 0


def test_discounted_objective_matches_exported_rate_and_outage(
    gurobi_license: None,
) -> None:
    config = make_config("current", discount_gamma=0.9)
    solver = HandoverOptimizerMILP(config)
    solver.load_data(make_episode(), ue_idx=0)
    solver.setup_model()
    solver.solve()
    arrays = solver.extract_results()
    weights = config.discount_gamma ** np.arange(solver.data.n_steps)
    expected = np.dot(weights, arrays["r_rate"] - config.lambda_r *
                      (arrays["i_ho_exec_running"] + arrays["i_rlf_running"])) / weights.sum()
    assert solver.model.ObjVal == pytest.approx(expected, abs=1e-6)
