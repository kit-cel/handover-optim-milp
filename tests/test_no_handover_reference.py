"""The standalone no-handover reference retains the radio link state machine."""

import numpy as np

from ho_optim_milp.reference.reference import RRCReferenceSimulation
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
    a3 = RRCReferenceSimulation(config, episode)
    no_handover = RRCReferenceSimulation(config, episode, trigger_handovers=False)
    a3.run()
    no_handover.run()

    assert a3.get_per_ue_result_metrics()[0]["num_ho"] == 1
    assert no_handover.get_per_ue_result_metrics()[0]["num_ho"] == 0
    assert no_handover.ue_list[0].get_flags()["prep_start"].sum() == 0
    np.testing.assert_array_equal(a3.sinr_db, no_handover.sinr_db)


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
    reference = RRCReferenceSimulation(
        make_config().rrc, episode, trigger_handovers=False
    )
    reference.run()
    metrics = reference.get_per_ue_result_metrics()[0]
    flags = reference.ue_list[0].get_flags()

    assert metrics["num_rlf"] > 0
    assert metrics["num_rlf"] == int(flags["rlf_no_ho"].sum())
    assert metrics["num_ho"] == int(flags["ho_success"].sum()) == 0
