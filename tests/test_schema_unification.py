"""One name per concept, across every method.

MILP rows and reference rows land in one table. Where they describe the
same quantity they must spell it the same way and encode "undefined" the
same way, or a merged release carries two conventions for one column and
every downstream join or average silently means something different than
it looks like.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ho_optim_milp.common.metrics import CAUSE_TO_COLUMN, build_metric_row
from ho_optim_milp.common.schema import (
    A3_PARAMETER_COLUMNS,
    HANDOVER_FAILURE_COUNT_COLUMNS,
    LINK_FAILURE_COUNT_COLUMNS,
    METRIC_COLUMNS,
    MILP_UE_COLUMN,
    STEP_SHARED_COLUMNS,
)
from ho_optim_milp.common.trajectory import milp_step_frame, reference_step_frame
from ho_optim_milp.common.metrics import LINK_FAILURE_CAUSES


def metric_row(**overrides):
    fields = dict(
        ue_id=3,
        simulated_time_s=60.0,
        simulated_steps=1500,
        mean_capacity=5.8,
        max_mean_capacity=6.4,
        rel_connected_time=0.98,
        num_ho=12,
        num_pp=1,
        num_rlf=2,
        link_failure_counts={"rlf_no_ho": 1, "hof_prep": 1, "hof_cmd": 0, "hof_exec": 0},
        objective_value=5.78,
    )
    fields.update(overrides)
    return build_metric_row(**fields)


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def test_the_ue_column_is_ue_id_everywhere() -> None:
    """``ue_idx`` was the reference side's spelling; it must be gone."""
    row = metric_row()
    assert MILP_UE_COLUMN == "ue_id"
    assert "ue_id" in row
    assert "ue_idx" not in row


def test_a3_parameter_columns_are_unprefixed() -> None:
    """Matches the released lambda_winners.csv that handover_rl reads.

    An ``a3_``-prefixed spelling would be a third name for the same three
    parameters, alongside the config field and the released column.
    """
    assert A3_PARAMETER_COLUMNS == ("ttt_ms", "hys_db", "offset_db")


def test_every_method_emits_the_same_metric_columns() -> None:
    """No method-dependent holes in the merged table."""
    with_causes = metric_row()
    without_causes = metric_row(link_failure_counts=None, num_rlf=2)

    assert set(METRIC_COLUMNS) <= set(with_causes)
    assert set(with_causes) == set(without_causes)


# --------------------------------------------------------------------------
# Undefined values
# --------------------------------------------------------------------------


def test_an_undefined_ratio_is_nan_not_inf() -> None:
    """The MILP side used to report ``inf`` where the reference used NaN.

    Averaging a column that mixes both conventions produces ``inf`` or
    ``nan`` depending on which rows happened to be included.
    """
    row = metric_row(num_ho=0, num_pp=0)
    assert math.isnan(row["pp_rate"])
    assert not math.isinf(row["pp_rate"])


def test_hof_rate_is_undefined_when_no_handover_was_attempted() -> None:
    """Zero would claim a perfect success rate for a UE that never tried."""
    row = metric_row(
        num_ho=0,
        num_rlf=0,
        link_failure_counts={c: 0 for c in LINK_FAILURE_CAUSES},
    )
    assert math.isnan(row["hof_rate"])


def test_rates_are_defined_when_the_denominator_is_positive() -> None:
    row = metric_row(num_ho=12, num_pp=3)
    assert row["pp_rate"] == pytest.approx(3 / 12)
    assert row["ho_per_s"] == pytest.approx(12 / 60.0)


# --------------------------------------------------------------------------
# The link-failure breakdown
# --------------------------------------------------------------------------


def test_cause_counts_sum_to_the_rlf_count() -> None:
    """The identity that makes the breakdown trustworthy."""
    row = metric_row()
    assert sum(row[column] for column in LINK_FAILURE_COUNT_COLUMNS) == row["num_rlf"]


def test_handover_failure_counts_sum_to_num_hof() -> None:
    row = metric_row(
        num_rlf=5,
        link_failure_counts={
            "rlf_no_ho": 2,
            "hof_prep": 1,
            "hof_cmd": 0,
            "hof_exec": 2,
        },
    )
    assert row["num_hof"] == 3
    assert sum(row[c] for c in HANDOVER_FAILURE_COUNT_COLUMNS) == row["num_hof"]


def test_inconsistent_cause_counts_are_rejected() -> None:
    """A breakdown that does not add up would be worse than none."""
    with pytest.raises(ValueError, match="sum to"):
        metric_row(
            num_rlf=5,
            link_failure_counts={c: 0 for c in LINK_FAILURE_CAUSES},
        )


def test_the_milp_reports_no_handover_failures() -> None:
    """Its formulation forbids a cell change while an RLF is recent.

    So every link failure it reports happened outside a handover. The
    columns still exist, reported as zero, so the schema is identical.
    """
    row = metric_row(link_failure_counts=None, num_rlf=4)
    assert row["num_hof"] == 0
    assert row["num_rlf_no_ho"] == 4
    assert all(row[c] == 0 for c in HANDOVER_FAILURE_COUNT_COLUMNS)


def test_cause_columns_are_derived_from_the_simulator_flag_names() -> None:
    """One list, so a new cause cannot be added to only one of the two."""
    assert set(CAUSE_TO_COLUMN) == set(LINK_FAILURE_CAUSES)
    assert set(CAUSE_TO_COLUMN.values()) == set(LINK_FAILURE_COUNT_COLUMNS)


# --------------------------------------------------------------------------
# Per-step trajectories
# --------------------------------------------------------------------------


def fake_milp_results(n_steps: int = 12, n_cells: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    pcell = rng.integers(0, n_cells, size=n_steps)
    sinr = rng.uniform(-5.0, 20.0, size=(n_cells, n_steps)).astype(np.float32)
    zeros = np.zeros(n_steps, dtype=np.int8)
    results = {
        "pcell": pcell,
        "sinr_db": sinr,
        "capacity": np.log2(1.0 + 10.0 ** (sinr / 10.0)),
        "i_ho_exec_running": zeros.copy(),
        "i_ho_exec_end": zeros.copy(),
        "i_rlf_running": zeros.copy(),
        "i_rlf_start": zeros.copy(),
        "x_bs": np.zeros((n_cells, n_steps), dtype=np.int8),
        "i_ho_same_cell": np.zeros((n_cells, n_steps), dtype=np.int8),
    }
    results["i_ho_exec_running"][3:5] = 1
    for name in (
        "i_q_in",
        "u_n310_raw",
        "c_n310",
        "tau_t310_rem",
        "i_t310_active",
        "s_rate",
        "r_rate",
    ):
        results[name] = np.zeros(n_steps)
    return results


def fake_reference_flags(n_steps: int = 12, n_ue: int = 2):
    flags = {
        name: np.zeros((n_steps, n_ue), dtype=np.int8)
        for name in (
            "ho_exec_end",
            "ho_exec_running",
            "hof_detected",
            "rlf_start",
            "rlf_running",
            *LINK_FAILURE_CAUSES,
        )
    }
    flags["pcell"] = np.zeros((n_steps, n_ue), dtype=np.int64)
    flags["ho_exec_running"][3:5] = 1
    return flags


def test_both_methods_write_the_same_shared_step_columns() -> None:
    """A consumer must be able to read either without knowing which."""
    milp = milp_step_frame(fake_milp_results())
    sinr = np.zeros((3, 2, 12), dtype=np.float32)
    reference = reference_step_frame(fake_reference_flags(), sinr, ue_index=0)

    for column in STEP_SHARED_COLUMNS:
        assert column in milp.columns, column
        assert column in reference.columns, column
        assert milp[column].dtype == reference[column].dtype, column


def test_delivered_capacity_is_zeroed_exactly_during_outage() -> None:
    """Derived in one place, so the two methods cannot disagree."""
    frame = milp_step_frame(fake_milp_results())
    outage = frame["i_ho_exec_running"] == 1

    assert (frame.loc[outage, "delivered_capacity"] == 0.0).all()
    assert (frame.loc[outage, "i_connected"] == 0).all()
    assert (
        frame.loc[~outage, "delivered_capacity"]
        == frame.loc[~outage, "serving_capacity"]
    ).all()


def test_serving_sinr_follows_the_serving_cell() -> None:
    results = fake_milp_results()
    frame = milp_step_frame(results)
    expected = results["sinr_db"][results["pcell"], np.arange(len(frame))]
    assert np.allclose(frame["serving_sinr_db"].to_numpy(), expected, atol=1e-5)


def test_milp_frame_keeps_every_auxiliary_variable() -> None:
    """Without them a stored solution cannot be re-audited.

    The equation-by-equation audit recomputes each constraint from the
    solved model's own variables; dropping them would make that
    impossible to repeat on released data.
    """
    frame = milp_step_frame(fake_milp_results())
    for name in ("u_n310_raw", "c_n310", "tau_t310_rem", "i_t310_active", "r_rate"):
        assert name in frame.columns, name
    # Per-cell arrays are expanded to flat columns, not list-valued cells.
    assert "x_bs_0" in frame.columns
    assert "sinr_db_0" in frame.columns and "capacity_0" in frame.columns


def test_reference_frame_keeps_the_failure_cause_flags() -> None:
    """The breakdown must be reconstructible per step, not only in totals."""
    sinr = np.zeros((3, 2, 12), dtype=np.float32)
    frame = reference_step_frame(fake_reference_flags(), sinr, ue_index=0)
    for cause in LINK_FAILURE_CAUSES:
        assert f"i_{cause}" in frame.columns, cause


def test_reference_frame_selects_the_requested_ue() -> None:
    flags = fake_reference_flags()
    flags["pcell"][:, 1] = 2
    sinr = np.zeros((3, 2, 12), dtype=np.float32)

    assert (reference_step_frame(flags, sinr, ue_index=0)["pcell"] == 0).all()
    assert (reference_step_frame(flags, sinr, ue_index=1)["pcell"] == 2).all()
