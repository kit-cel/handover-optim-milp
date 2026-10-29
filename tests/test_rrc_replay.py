"""Stored handover completions replay through the simulator RRC."""

import numpy as np

from ho_optim_milp.dataset_generator.recorded_results import reference_recorded_episode
from ho_optim_milp.dataset_generator.rrc_replay import replay_stored_rrc_trajectory

from test_model_profiles import make_config
from test_recorded_results import _episode


def test_replay_reconstructs_a_completed_handover() -> None:
    episode = _episode()
    config = make_config().rrc
    _, stored = reference_recorded_episode(episode, config, lambda_r=0.0)[0]
    assert stored["i_ho_exec_end"].sum() == 1
    replayed = replay_stored_rrc_trajectory(
        config, episode, ue_id=0, stored_steps=stored, method="milp"
    )
    np.testing.assert_array_equal(replayed["pcell"], stored["pcell"])
    np.testing.assert_array_equal(replayed["i_ho_exec_end"], stored["i_ho_exec_end"])


def test_replay_keeps_no_handover_rlm_active() -> None:
    episode = _episode()
    config = make_config().rrc
    _, stored = reference_recorded_episode(
        episode, config, lambda_r=0.0, trigger_handovers=False
    )[0]
    replayed = replay_stored_rrc_trajectory(
        config, episode, ue_id=0, stored_steps=stored, method="no_ho"
    )
    assert replayed["i_ho_exec_end"].sum() == 0
    np.testing.assert_array_equal(replayed["pcell"], stored["pcell"])
