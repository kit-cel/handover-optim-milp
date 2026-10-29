"""No-handover replay retains radio-link monitoring in the simulator RRC."""

import numpy as np

from ho_optim_milp.dataset_generator.recorded_results import reference_recorded_episode
from ho_optim_milp.result_manager.episode_result import EpisodeResult

from test_model_profiles import make_config


def test_no_handover_suppresses_a3_without_changing_recorded_radio() -> None:
    sinr = np.asarray([[12, 0]] * 4 + [[0, 12]] * 16, dtype=np.float32)[:, None, :]
    episode = EpisodeResult(
        ep_name="short_handover",
        ue_pos=np.zeros((20, 1, 2), dtype=np.float32),
        distances_2d=np.zeros_like(sinr),
        distances_3d=np.zeros_like(sinr),
        rsrp=sinr - 70.0,
        sinr=sinr,
        ue_move_angle=np.zeros((20, 1), dtype=np.float32),
    )
    config = make_config().rrc
    a3_metrics, a3_steps = reference_recorded_episode(
        episode, config, lambda_r=0.0
    )[0]
    no_ho_metrics, no_ho_steps = reference_recorded_episode(
        episode, config, lambda_r=0.0, trigger_handovers=False
    )[0]

    assert a3_metrics["num_ho"] == 1
    assert no_ho_metrics["num_ho"] == 0
    assert no_ho_steps["i_prep_start"].sum() == 0
    np.testing.assert_array_equal(a3_steps["sinr_db_0"], no_ho_steps["sinr_db_0"])


def test_no_handover_still_detects_radio_link_failures() -> None:
    sinr = np.asarray(
        [[12, 0]] * 4 + [[-20, -20]] * 10 + [[12, 0]] * 6,
        dtype=np.float32,
    )[:, None, :]
    episode = EpisodeResult(
        ep_name="short_outage",
        ue_pos=np.zeros((20, 1, 2), dtype=np.float32),
        distances_2d=np.zeros_like(sinr),
        distances_3d=np.zeros_like(sinr),
        rsrp=sinr - 70.0,
        sinr=sinr,
        ue_move_angle=np.zeros((20, 1), dtype=np.float32),
    )
    metrics, steps = reference_recorded_episode(
        episode, make_config().rrc, lambda_r=0.0, trigger_handovers=False
    )[0]

    assert metrics["num_rlf"] > 0
    assert metrics["num_rlf"] == int(steps["i_rlf_no_ho"].sum())
    assert metrics["num_ho"] == int(steps["i_ho_success"].sum()) == 0
