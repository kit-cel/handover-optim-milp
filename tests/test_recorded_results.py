"""Canonical result rows from short stored radio traces."""

import numpy as np
import pytest

from ho_optim_milp.dataset_generator.recorded_results import reference_recorded_episode
from ho_optim_milp.result_manager.episode_result import EpisodeResult

from test_model_profiles import make_config


def _episode() -> EpisodeResult:
    sinr = np.asarray([[12, 0]] * 4 + [[0, 12]] * 16, dtype=np.float32)[:, None, :]
    return EpisodeResult(
        ep_name="short_reference",
        ue_pos=np.zeros((20, 1, 2), dtype=np.float32),
        distances_2d=np.zeros_like(sinr),
        distances_3d=np.zeros_like(sinr),
        rsrp=sinr - 70.0,
        sinr=sinr,
        ue_move_angle=np.zeros((20, 1), dtype=np.float32),
    )


@pytest.mark.parametrize("backend", ["standalone", "simulator"])
def test_reference_rows_and_steps_use_one_capacity_definition(backend: str) -> None:
    if backend == "simulator":
        pytest.importorskip("mobile_network_sim")
    config = make_config().rrc
    results = reference_recorded_episode(
        _episode(), config, lambda_r=1.0, backend=backend
    )
    metrics, steps = results[0]

    assert len(steps) == metrics["simulated_steps"] == 20
    assert metrics["num_ho"] == int(steps["i_ho_exec_end"].sum()) == 1
    assert metrics["num_rlf"] == sum(
        metrics[f"num_{cause}"] for cause in
        ("rlf_no_ho", "hof_prep", "hof_cmd", "hof_exec")
    )
    assert metrics["mean_capacity"] == pytest.approx(steps["delivered_capacity"].mean())
    assert metrics["ho_objective_value"] == pytest.approx(
        metrics["mean_capacity"] - (1.0 - metrics["rel_connected_time"])
    )
    assert {"l3_rsrp_dbm_0", "l1_qin_sinr_db_0", "q_in_0"} <= set(steps)


@pytest.mark.parametrize("backend", ["standalone", "simulator"])
def test_no_handover_keeps_radio_link_monitoring(backend: str) -> None:
    if backend == "simulator":
        pytest.importorskip("mobile_network_sim")
    metrics, steps = reference_recorded_episode(
        _episode(), make_config().rrc, lambda_r=1.0,
        backend=backend, trigger_handovers=False,
    )[0]
    assert metrics["num_ho"] == 0
    assert steps["i_prep_start"].sum() == 0
