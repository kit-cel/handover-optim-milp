"""A3 parameter selection at three levels of adaptation.

The load-bearing property is *where* each level is allowed to look. A
level that silently peeks at its evaluation split stops being a
baseline, and nothing about the resulting numbers would look wrong.
"""

from __future__ import annotations

import pytest

from ho_optim_milp.dataset_generator.a3_selection import (
    DEFAULT_SPEED_BINS,
    LEVEL_GLOBAL,
    LEVEL_SPEED_AWARE,
    LEVEL_SPLIT_ORACLE,
    A3Parameters,
    build_selection_table,
    clip_to_highest_bin,
    objective_score,
    select_global,
    select_speed_aware,
    select_split_oracle,
    speed_bin_label,
)


def row(
    *,
    split: str,
    ttt: float,
    hys: float,
    offset: float,
    capacity: float,
    connected: float,
    speed: float = 60.0,
) -> dict:
    return {
        "split": split,
        "ttt_ms": ttt,
        "hys_db": hys,
        "offset_db": offset,
        "mean_capacity": capacity,
        "rel_connected_time": connected,
        "mean_speed_kph": speed,
    }


def two_parameter_rows(split: str, *, speed: float = 60.0) -> list[dict]:
    """Two parameter sets whose ranking flips with lambda.

    At lambda=0 the higher-capacity set wins; at a large lambda the
    better-connected one does. That makes lambda-dependence testable
    rather than incidental.
    """
    return [
        row(
            split=split,
            ttt=160,
            hys=0.0,
            offset=3.0,
            capacity=5.80,
            connected=0.960,
            speed=speed,
        ),
        row(
            split=split,
            ttt=320,
            hys=1.0,
            offset=2.0,
            capacity=5.70,
            connected=0.985,
            speed=speed,
        ),
    ]


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------


def test_objective_is_the_papers_equation_7() -> None:
    assert objective_score(5.8, 0.96, 0.0) == pytest.approx(5.8)
    assert objective_score(5.8, 0.96, 10.0) == pytest.approx(5.8 - 10.0 * 0.04)


def test_selection_follows_lambda() -> None:
    """The same results must yield different winners at different lambdas."""
    rows = two_parameter_rows("id_val")

    at_zero = select_global(rows, lambda_value=0.0, selection_split="id_val")
    at_high = select_global(rows, lambda_value=10.0, selection_split="id_val")

    assert at_zero.parameters == A3Parameters(160.0, 0.0, 3.0)
    assert at_high.parameters == A3Parameters(320.0, 1.0, 2.0)


# --------------------------------------------------------------------------
# Where each level is allowed to look
# --------------------------------------------------------------------------


def test_global_selection_ignores_rows_from_other_splits() -> None:
    """Test-split rows must not influence the deployable baseline.

    The decoy below scores far better than anything on the selection
    split; if it were considered, it would win.
    """
    rows = two_parameter_rows("id_val") + [
        row(
            split="id_test",
            ttt=999,
            hys=9.0,
            offset=9.0,
            capacity=99.0,
            connected=1.0,
        )
    ]
    selection = select_global(rows, lambda_value=0.0, selection_split="id_val")

    assert selection.parameters.ttt_ms != 999
    assert selection.selected_on == "id_val"
    assert selection.deployable


def test_global_selection_refuses_when_the_selection_split_is_absent() -> None:
    """Silently selecting on whatever was passed in is the failure mode."""
    rows = two_parameter_rows("id_test")
    with pytest.raises(ValueError, match="no A3 rows from selection split"):
        select_global(rows, lambda_value=0.0, selection_split="id_val")


def test_split_oracle_is_marked_non_deployable() -> None:
    """It uses the evaluation split's own results and must say so."""
    rows = two_parameter_rows("id_test")
    selection = select_split_oracle(rows, lambda_value=0.0, split="id_test")

    assert selection.level == LEVEL_SPLIT_ORACLE
    assert selection.selected_on == "id_test"
    assert not selection.deployable


def test_split_oracle_scores_at_least_as_well_as_the_frozen_global() -> None:
    """Sanity check on what the oracle means.

    Re-optimizing on the evaluation split cannot do worse than a set
    frozen elsewhere; if it did, the selection logic would be wrong.
    """
    id_val = two_parameter_rows("id_val")
    # On id_test the ranking is reversed, so the frozen choice is not optimal.
    id_test = [
        row(split="id_test", ttt=160, hys=0.0, offset=3.0, capacity=5.60, connected=0.95),
        row(split="id_test", ttt=320, hys=1.0, offset=2.0, capacity=5.90, connected=0.99),
    ]
    frozen = select_global(id_val, lambda_value=0.0, selection_split="id_val")
    oracle = select_split_oracle(id_test, lambda_value=0.0, split="id_test")

    frozen_on_test = next(
        objective_score(r["mean_capacity"], r["rel_connected_time"], 0.0)
        for r in id_test
        if r["ttt_ms"] == frozen.parameters.ttt_ms
    )
    assert oracle.score >= frozen_on_test


# --------------------------------------------------------------------------
# Speed-aware level
# --------------------------------------------------------------------------


def test_speed_bins_are_half_open_except_the_last() -> None:
    assert speed_bin_label(30.0, DEFAULT_SPEED_BINS) == "[30,50)"
    assert speed_bin_label(49.9, DEFAULT_SPEED_BINS) == "[30,50)"
    assert speed_bin_label(50.0, DEFAULT_SPEED_BINS) == "[50,70)"
    assert speed_bin_label(70.0, DEFAULT_SPEED_BINS) == "[70,90]"
    # The declared range's upper endpoint must be covered.
    assert speed_bin_label(90.0, DEFAULT_SPEED_BINS) == "[70,90]"
    # Outside every bin -- e.g. an OOD-Speed UE.
    assert speed_bin_label(120.0, DEFAULT_SPEED_BINS) is None


def test_speed_aware_selects_one_set_per_bin() -> None:
    rows = (
        two_parameter_rows("id_val", speed=40.0)
        + two_parameter_rows("id_val", speed=60.0)
        + two_parameter_rows("id_val", speed=80.0)
    )
    selections = select_speed_aware(
        rows, lambda_value=0.0, selection_split="id_val"
    )

    assert len(selections) == 3
    assert {s.bin_label for s in selections} == {"[30,50)", "[50,70)", "[70,90]"}
    assert all(s.level == LEVEL_SPEED_AWARE for s in selections)
    assert all(s.selected_on == "id_val" for s in selections)


def test_speed_aware_refuses_an_empty_bin() -> None:
    """A bin with no data cannot be given a parameter set."""
    rows = two_parameter_rows("id_val", speed=40.0)
    with pytest.raises(ValueError, match="speed bins with no results"):
        select_speed_aware(rows, lambda_value=0.0, selection_split="id_val")


def test_ood_speed_clips_onto_the_highest_bin_without_retuning() -> None:
    """The speed-aware level must never see OOD data.

    OOD-Speed (90-130 km/h) is above every ID bin, so it reuses the top
    bin's parameters rather than getting its own.
    """
    rows = (
        two_parameter_rows("id_val", speed=40.0)
        + two_parameter_rows("id_val", speed=60.0)
        + two_parameter_rows("id_val", speed=80.0)
    )
    selections = select_speed_aware(rows, lambda_value=0.0, selection_split="id_val")
    clipped = clip_to_highest_bin(selections)

    assert clipped.bin_label == "[70,90]"
    assert clipped.selected_on == "id_val"
    # An OOD speed genuinely falls outside every bin, which is why
    # clipping (rather than binning) is the right operation.
    assert speed_bin_label(110.0, DEFAULT_SPEED_BINS) is None


# --------------------------------------------------------------------------
# Aggregation and determinism
# --------------------------------------------------------------------------


def test_scores_are_averaged_over_a_parameter_sets_rows() -> None:
    """A parameter set is judged on its average, not its luckiest UE."""
    rows = [
        row(split="id_val", ttt=160, hys=0.0, offset=3.0, capacity=4.0, connected=1.0),
        row(split="id_val", ttt=160, hys=0.0, offset=3.0, capacity=6.0, connected=1.0),
        row(split="id_val", ttt=320, hys=0.0, offset=3.0, capacity=5.5, connected=1.0),
    ]
    selection = select_global(rows, lambda_value=0.0, selection_split="id_val")

    # Mean of the first set is 5.0, below the second set's single 5.5,
    # even though its best individual row (6.0) is higher.
    assert selection.parameters.ttt_ms == 320
    assert selection.score == pytest.approx(5.5)


def test_ties_are_broken_reproducibly_regardless_of_row_order() -> None:
    rows = [
        row(split="id_val", ttt=320, hys=1.0, offset=2.0, capacity=5.0, connected=1.0),
        row(split="id_val", ttt=160, hys=0.0, offset=3.0, capacity=5.0, connected=1.0),
    ]
    forward = select_global(rows, lambda_value=0.0, selection_split="id_val")
    backward = select_global(
        list(reversed(rows)), lambda_value=0.0, selection_split="id_val"
    )
    assert forward.parameters == backward.parameters


def test_selecting_from_nothing_raises() -> None:
    with pytest.raises(ValueError, match="no A3 rows"):
        select_split_oracle([], lambda_value=0.0, split="id_test")


# --------------------------------------------------------------------------
# The full reported table
# --------------------------------------------------------------------------


def test_selection_table_labels_every_row_with_its_provenance() -> None:
    """A reader must never mistake the oracle for a deployable baseline."""
    rows = (
        two_parameter_rows("id_val", speed=40.0)
        + two_parameter_rows("id_val", speed=60.0)
        + two_parameter_rows("id_val", speed=80.0)
        + two_parameter_rows("id_test", speed=60.0)
        + two_parameter_rows("ood_speed", speed=110.0)
    )
    table = build_selection_table(
        rows,
        lambda_values=[0.0, 10.0],
        selection_split="id_val",
        evaluation_splits=["id_test", "ood_speed"],
    )

    levels = {entry["level"] for entry in table}
    assert levels == {LEVEL_GLOBAL, LEVEL_SPEED_AWARE, LEVEL_SPLIT_ORACLE}
    assert all("deployable" in entry and "selected_on" in entry for entry in table)

    for entry in table:
        if entry["level"] == LEVEL_SPLIT_ORACLE:
            assert not entry["deployable"]
            assert entry["selected_on"] == entry["evaluated_on"]
        else:
            assert entry["deployable"]
            assert entry["selected_on"] == "id_val"


def test_selection_table_covers_every_lambda() -> None:
    rows = (
        two_parameter_rows("id_val", speed=40.0)
        + two_parameter_rows("id_val", speed=60.0)
        + two_parameter_rows("id_val", speed=80.0)
        + two_parameter_rows("id_test", speed=60.0)
    )
    lambdas = [0.0, 0.1, 10.0, 1000.0]
    table = build_selection_table(
        rows,
        lambda_values=lambdas,
        selection_split="id_val",
        evaluation_splits=["id_test"],
    )
    assert {entry["lambda"] for entry in table} == set(lambdas)
    # per lambda: 1 global + 3 speed bins + 1 oracle
    assert len(table) == len(lambdas) * 5
