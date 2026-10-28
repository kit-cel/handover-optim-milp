"""Numerical regression cases adapted from the simulator-side MILP tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from ho_optim_milp.common.metrics import count_ping_pongs
from ho_optim_milp.dataloader.cell_ranking import rank1_neighbor_table
from ho_optim_milp.dataloader.preprocessing import preprocess_dataset
from ho_optim_milp.optimization.milp_dataclasses import ProblemData, RRCConstants
from ho_optim_milp.optimization.milp_result_extractor import ResultExtractor

from test_model_profiles import make_config, make_episode


def test_current_inputs_are_raw_but_rlm_uses_l1_filter() -> None:
    episode = make_episode()
    rrc = make_config().rrc
    raw_rsrp, raw_sinr, q_in, q_out = preprocess_dataset(
        episode, rrc, use_l3_filtering=False, ue_no=0
    )
    np.testing.assert_array_equal(raw_sinr, episode.sinr[:, 0, :].T)
    np.testing.assert_array_equal(raw_rsrp, episode.rsrp[:, 0, :].T)
    assert q_in.shape == q_out.shape == (2, 8)
    # At the third sample the two-sample L1 average is applied per cell.
    expected = (episode.sinr[1, 0, :] + episode.sinr[2, 0, :]) / 2
    np.testing.assert_array_equal(q_in[:, 2], expected >= rrc.q_in_db)
    np.testing.assert_array_equal(q_out[:, 2], expected <= rrc.q_out_db)
    _, filtered_sinr, _, _ = preprocess_dataset(
        episode, rrc, use_l3_filtering=True, ue_no=0
    )
    assert not np.array_equal(filtered_sinr, raw_sinr)


def test_rank1_neighbor_excludes_current_cell_and_breaks_ties_stably() -> None:
    rsrp = np.array([[-80.0, -70.0], [-80.0, -75.0], [-90.0, -65.0]])
    targets = rank1_neighbor_table(rsrp)
    np.testing.assert_array_equal(targets[:, 0], [1, 0, 0])
    np.testing.assert_array_equal(targets[:, 1], [2, 2, 0])
    np.testing.assert_array_equal(rank1_neighbor_table(np.ones((1, 2))), [[-1, -1]])


def test_ping_pong_requires_completed_return_within_window() -> None:
    pcell = np.array([0, 0, 1, 1, 0, 0, 1], dtype=np.int32)
    completed = np.array([0, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    assert count_ping_pongs(pcell, completed, 2) == 2
    assert count_ping_pongs(pcell, completed, 1) == 0


def _variable(values: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(X=values)


def _counter(n_steps: int) -> SimpleNamespace:
    zeros = np.zeros(n_steps)
    return SimpleNamespace(**{
        key: _variable(zeros.copy())
        for key in ("u_raw", "u_sat", "cnt", "ind", "b_reset", "b_q", "b_c")
    })


def test_result_export_rounding_ho_count_and_ping_pong(tmp_path) -> None:
    n_steps = 5
    pcell = np.array([0, 0, 1, 1, 0])
    x_bs = np.eye(2)[pcell].T.astype(float)
    x_bs[x_bs == 1] = 0.99999
    x_bs[x_bs == 0] = 0.00001
    ho_end = np.array([0, 0, 0.99999, 0, 0.99999])
    zeros = np.zeros(n_steps)
    variables = SimpleNamespace(
        x_bs=_variable(x_bs), i_q_in=_variable(zeros), i_q_out=_variable(zeros),
        n310=_counter(n_steps), n311=_counter(n_steps),
        t310=SimpleNamespace(**{
            key: _variable(zeros.copy()) for key in
            ("start", "stop", "active", "cancel_due_to_n311", "expiry", "tau",
             "tau_eq_1", "b_mode_rlf", "b_mode_start", "b_mode_stop", "b_mode_decr")
        }),
        rlf=SimpleNamespace(start=_variable(zeros), running=_variable(zeros),
                            in_last_t_ho_steps=_variable(zeros)),
        ho=SimpleNamespace(same_cell=_variable(x_bs), cell_change=_variable(zeros),
                           exec_end=_variable(ho_end), exec_running=_variable(zeros)),
        rate=SimpleNamespace(conn_aux=_variable(zeros), s_rate=_variable(zeros),
                             r_rate=_variable(zeros)),
    )
    data = ProblemData(0, 2, n_steps, np.zeros((2, n_steps)),
                       np.ones((2, n_steps)), np.ones((2, n_steps)),
                       np.zeros((2, n_steps)))
    consts = RRCConstants(40, -6.0, -8.0, 1, 10, 3, 1, 1, 2, 3)
    extractor = ResultExtractor(variables, data, consts, simulation_id="test",
                                lambda_r=1.0, ep_name="ep_0000", log_dir=str(tmp_path))
    arrays = extractor.extract_results()
    metrics = extractor.get_result_metrics(0.0)[0]
    assert metrics["num_ho"] == 2
    assert metrics["num_pp"] == 1
    assert arrays["i_ho_exec_end"].tolist() == [0, 0, 1, 0, 1]
    assert arrays["x_bs"].sum(axis=0).tolist() == [1] * n_steps
    assert "i_t310_eq_1" in arrays
    assert "i_rate_conn_aux" in arrays
    assert extractor._get_snapshot(2)["i_ho_exec_end"] == 1
    with pytest.raises(ValueError, match="out of bounds"):
        extractor._get_snapshot(n_steps)
