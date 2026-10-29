from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from ho_optim_milp.dataset_generator.events import (
    derive_handover_attempts,
    derive_outage_decomposition,
    derive_rrc_events,
)


def _steps(*, outcome: str = "success") -> pd.DataFrame:
    n = 9
    frame = pd.DataFrame(
        {
            "release_id": ["sample_release"] * n,
            "split": ["id_test"] * n,
            "method": ["a3"] * n,
            "selection_level": ["global"] * n,
            "lambda": [1.0] * n,
            "episode_idx": [0] * n,
            "ue_id": [0] * n,
            "trajectory_id": ["a3:id_test:global:1:0:0"] * n,
            "selected_on": ["id_val"] * n,
            "evaluated_on": ["id_test"] * n,
            "deployable": [True] * n,
            "bin_label": [None] * n,
            "ttt_ms": [40.0] * n,
            "hys_db": [1.0] * n,
            "offset_db": [0.0] * n,
            "time_step": list(range(n)),
            "pcell": [0, 0, 0, 0, 0, 0, 1 if outcome == "success" else 0, 0, 0],
            "target_cell": [None, 1, 1, 1, 1, 1, 1, None, None],
            "i_connected": [
                1,
                1,
                1,
                1,
                0,
                0,
                0 if outcome != "success" else 1,
                0 if outcome != "success" else 1,
                1,
            ],
            "i_ho_exec_running": [0, 0, 0, 0, 1, 1, 0, 0, 0],
            "i_rlf_running": [
                0,
                0,
                0,
                0,
                0,
                0,
                1 if outcome != "success" else 0,
                1 if outcome != "success" else 0,
                0,
            ],
            "i_ttt_start": [0, 1, 0, 0, 0, 0, 0, 0, 0],
            "i_ttt_expire": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "i_prep_start": [0, 0, 1, 0, 0, 0, 0, 0, 0],
            "i_prep_expire": [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "i_exec_start": [0, 0, 0, 0, 1, 0, 0, 0, 0],
            "i_exec_expire": [0, 0, 0, 0, 0, 0, 1, 0, 0],
            "i_ho_success": [0, 0, 0, 0, 0, 0, 1 if outcome == "success" else 0, 0, 0],
            "i_hof_exec": [0, 0, 0, 0, 0, 0, 1 if outcome == "hof_exec" else 0, 0, 0],
            "i_rlf_start": [0, 0, 0, 0, 0, 0, 1 if outcome != "success" else 0, 0, 0],
            "rlm_evaluated": [True, True, True, True, False, False, False, False, True],
        }
    )
    for prefix in (
        "sinr_db",
        "l3_rsrp_dbm",
        "l3_sinr_db",
        "l1_qin_sinr_db",
        "l1_qout_sinr_db",
    ):
        frame[f"{prefix}_0"] = range(n)
        frame[f"{prefix}_1"] = [100 + value for value in range(n)]
    return frame


def test_attempt_uses_current_boundary_signals_and_keeps_outcome_global() -> None:
    attempts = derive_handover_attempts(_steps(outcome="hof_exec"), t_res_ms=40)
    assert len(attempts) == 1
    attempt = attempts.iloc[0]
    assert attempt["outcome"] == "hof_exec"
    assert attempt["prep_end_step"] == 4
    assert attempt["exec_end_step"] == 6
    assert attempt["prep_end_l1_qout_sinr_db"] == 4
    assert attempt["exec_end_l1_qout_sinr_db"] == 106
    assert attempt["prep_duration_s"] == 0.08
    assert attempt["exec_duration_s"] == 0.08


def test_nullable_attempt_fields_have_stable_arrow_types() -> None:
    import pyarrow as pa

    success = derive_handover_attempts(_steps(), t_res_ms=40)
    failed = derive_handover_attempts(_steps(outcome="hof_exec"), t_res_ms=40)

    success_schema = pa.Table.from_pandas(success, preserve_index=False).schema
    failed_schema = pa.Table.from_pandas(failed, preserve_index=False).schema
    assert success_schema == failed_schema
    assert not pa.types.is_null(success_schema.field("failure_cause").type)
    assert not pa.types.is_null(success_schema.field("censored_phase").type)


def test_success_can_share_sample_with_next_zero_ttt_prep_start() -> None:
    steps = _steps()
    boundary = 6
    steps.loc[boundary, "target_cell"] = 0
    steps.loc[boundary, "i_ttt_start"] = 1
    steps.loc[boundary, "i_ttt_expire"] = 1
    steps.loc[boundary, "i_prep_start"] = 1

    attempts = derive_handover_attempts(steps, t_res_ms=40)

    assert list(attempts["outcome"]) == ["success", "censored"]
    completed, started = attempts.iloc[0], attempts.iloc[1]
    assert completed["exec_end_step"] == boundary
    assert completed["target_cell"] == 1
    assert completed["exec_end_l1_qout_sinr_db"] == 106
    assert started["prep_start_step"] == boundary
    assert started["source_cell"] == 1
    assert started["target_cell"] == 0
    assert started["ttt_start_step"] == boundary
    assert started["ttt_expire_step"] == boundary


@pytest.mark.filterwarnings("ignore:CHO_IDEALIZED:RuntimeWarning")
@pytest.mark.parametrize("n_steps", [5, 6])
def test_zero_preparation_attempts_keep_same_sample_events_separate(n_steps) -> None:
    simulator_rrc = pytest.importorskip("mobile_network_sim.rrc")
    cfg = simulator_rrc.RRCConfig(
        t_res_ms=40,
        phy_update_interval_ms=40,
        msr_interval_ms=40,
        ho_event_config={"a3": {"ttt_ms": 0, "hys": 0.0, "offset": 0.0}},
        handover_mode="cho_idealized",
        t_ho_prep_ms_simulated=0,
        t_ho_exec_ms_simulated=80,
        l3_filter_coef=0,
    )
    ue = simulator_rrc.RRCStepUE(0, cfg)
    ue.set_learning_mode(True, True)
    rows = []
    for t in range(n_steps):
        ue.step(
            time_ms=t * 40,
            rsrp_dbm=np.asarray([-60.0, -70.0]),
            sinr_db=np.asarray([10.0, 20.0]),
            action={1: 2, 3: 1}.get(t, 0),
            action_mode="target_cell",
        )
        rows.append({"time_step": t, **ue.snapshot()})
    steps = pd.DataFrame(rows)
    attempts = derive_handover_attempts(steps, t_res_ms=40)
    assert attempts["prep_start_step"].tolist() == [1, 3]
    assert attempts["prep_end_step"].tolist() == [1, 3]
    assert attempts["exec_start_step"].tolist() == [1, 3]
    assert attempts["prep_duration_samples"].tolist() == [0, 0]
    assert attempts["source_cell"].tolist() == [0, 1]
    assert attempts["target_cell"].tolist() == [1, 0]
    assert attempts.iloc[0]["exec_end_step"] == 3
    assert attempts.iloc[0]["exec_duration_s"] == 0.08
    if n_steps == 6:
        assert attempts["outcome"].tolist() == ["success", "success"]
        assert attempts["exec_duration_samples"].tolist() == [2, 2]
        assert attempts.iloc[1]["exec_end_step"] == 5
    else:
        assert attempts["outcome"].tolist() == ["success", "censored"]
        assert attempts.iloc[1]["censored_phase"] == "exec"
        assert pd.isna(attempts.iloc[1]["exec_end_step"])


def test_events_are_long_form_and_include_phase_boundaries() -> None:
    events = derive_rrc_events(_steps(), t_res_ms=40)
    assert {"prep_start", "prep_end", "exec_start", "exec_end", "ho_success"} <= set(
        events["event_type"]
    )
    assert events["time_ms"].eq(events["time_step"] * 40).all()


def test_outage_decomposition_is_disjoint_for_success_and_hof() -> None:
    success_steps = _steps()
    success_attempts = derive_handover_attempts(success_steps, t_res_ms=40)
    success = derive_outage_decomposition(
        success_steps, success_attempts, t_res_ms=40
    ).iloc[0]
    assert success["successful_ho_exec_samples"] == 2
    assert success["outage_samples"] == 2

    hof_steps = _steps(outcome="hof_exec")
    hof_attempts = derive_handover_attempts(hof_steps, t_res_ms=40)
    hof = derive_outage_decomposition(hof_steps, hof_attempts, t_res_ms=40).iloc[0]
    assert hof["hof_related_samples"] == 4
    assert hof["rlf_no_ho_samples"] == 0
    assert hof["unclassified_or_censored_samples"] == 0
    assert hof["outage_samples"] == 4


def test_no_handover_has_empty_attempt_schema() -> None:
    frame = _steps().assign(
        method="no_ho",
        selection_level=None,
        i_ttt_start=0,
        i_ttt_expire=0,
        i_prep_start=0,
        i_prep_expire=0,
        i_exec_start=0,
        i_exec_expire=0,
        i_ho_success=0,
    )
    attempts = derive_handover_attempts(frame, t_res_ms=40)
    assert attempts.empty
    assert {"attempt_id", "outcome", "prep_end_l1_qin_sinr_db"} <= set(attempts.columns)
