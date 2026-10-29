"""Selecting A3 reference parameters at three levels of adaptation.

A single "best A3" number is not a fair baseline: how well A3 does
depends entirely on how much it was allowed to adapt to the data it is
scored on. The release therefore reports three explicitly labelled
levels, and the labelling is what makes them interpretable:

``global``
    One parameter set chosen on ID-Val across all speeds, then frozen
    for ID-Test and both OOD splits. This is the *deployable* baseline
    and the one the primary hypothesis is tested against.

``speed_aware``
    One parameter set per predefined ID speed bin, still chosen only on
    ID-Val. OOD-Speed is clipped onto the highest bin without retuning,
    because a deployed network cannot retune for a speed regime it has
    never seen.

``split_oracle``
    Re-optimized separately on each test split. **Not deployable** --
    it uses the test split's own labels. It exists as an upper bound on
    what any A3 parameterization could achieve, so a gap to it says
    something different from a gap to ``global``.

Selection always uses the same objective as the MILP and the agents,
paper Eq. (7): ``s_k(lambda) = L_r,k - lambda * (1 - L_c,k)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import A3_HYS_COLUMN, A3_OFFSET_COLUMN, A3_TTT_COLUMN

LEVEL_GLOBAL = "global"
LEVEL_SPEED_AWARE = "speed_aware"
LEVEL_SPLIT_ORACLE = "split_oracle"
LEVELS = (LEVEL_GLOBAL, LEVEL_SPEED_AWARE, LEVEL_SPLIT_ORACLE)

# Speed bins for the speed-aware level, fixed in advance so the binning
# is not itself tuned on results. Half-open except the last.
DEFAULT_SPEED_BINS: tuple[tuple[float, float], ...] = (
    (30.0, 50.0),
    (50.0, 70.0),
    (70.0, 90.0),
)


@dataclass(frozen=True, slots=True, order=True)
class A3Parameters:
    """One A3 parameter set.

    Ordered so ties in the selection score can be broken by the
    parameters themselves rather than by dict iteration order, which
    would make the winner depend on how the results were read off disk.
    """

    ttt_ms: float
    hys_db: float
    offset_db: float

    def as_row(self) -> dict[str, float]:
        """Return the parameters as flat columns."""
        return {
            A3_TTT_COLUMN: self.ttt_ms,
            A3_HYS_COLUMN: self.hys_db,
            A3_OFFSET_COLUMN: self.offset_db,
        }


@dataclass(frozen=True, slots=True)
class A3Selection:
    """The chosen parameters for one level, with what they were chosen on."""

    level: str
    lambda_value: float
    parameters: A3Parameters
    selected_on: str
    """Split the selection was made on -- the auditable part."""
    score: float
    bin_label: str | None = None
    deployable: bool = True

    def as_row(self) -> dict[str, Any]:
        """Return a flat result row for the release manifest."""
        return {
            "level": self.level,
            "lambda": self.lambda_value,
            "selected_on": self.selected_on,
            "score": self.score,
            "bin_label": self.bin_label,
            "deployable": self.deployable,
            **self.parameters.as_row(),
        }


def objective_score(
    mean_capacity: float, rel_connected_time: float, lambda_value: float
) -> float:
    """Paper Eq. (7): the same rate/connectivity trade-off as Eq. (4)."""
    return mean_capacity - lambda_value * (1.0 - rel_connected_time)


def _aggregate_by_parameters(
    rows: Iterable[Mapping[str, Any]], lambda_value: float
) -> dict[A3Parameters, float]:
    """Average each parameter set's score over the rows it appears in.

    Averaging the *score* per parameter set, rather than picking the best
    single trajectory, is what makes the choice a property of the
    parameter set instead of a lucky UE.
    """
    totals: dict[A3Parameters, list[float]] = {}
    for row in rows:
        if row.get("method") not in (None, "a3"):
            continue
        if "lambda" in row and float(row["lambda"]) != float(lambda_value):
            continue
        parameters = A3Parameters(
            ttt_ms=float(row[A3_TTT_COLUMN]),
            hys_db=float(row[A3_HYS_COLUMN]),
            offset_db=float(row[A3_OFFSET_COLUMN]),
        )
        totals.setdefault(parameters, []).append(
            objective_score(
                float(row["mean_capacity"]),
                float(row["rel_connected_time"]),
                lambda_value,
            )
        )
    return {
        parameters: sum(scores) / len(scores) for parameters, scores in totals.items()
    }


def _best(scored: Mapping[A3Parameters, float]) -> tuple[A3Parameters, float]:
    """Return the highest-scoring parameter set, breaking ties reproducibly.

    Ties are broken by the parameters themselves rather than by dict
    order, so the same input always yields the same winner regardless of
    how the rows happened to be ordered on disk.
    """
    if not scored:
        raise ValueError("no A3 results to select from")
    best_score = max(scored.values())
    candidates = sorted(
        parameters for parameters, score in scored.items() if score == best_score
    )
    return candidates[0], best_score


def _eligible_rows(
    rows: Sequence[Mapping[str, Any]], *, split: str, lambda_value: float
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("split") == split
        and row.get("method") in (None, "a3")
        and ("lambda" not in row or float(row["lambda"]) == float(lambda_value))
    ]


def select_global(
    rows: Sequence[Mapping[str, Any]],
    *,
    lambda_value: float,
    selection_split: str,
) -> A3Selection:
    """Choose one deployable parameter set on the selection split.

    Parameters
    ----------
    rows
        A3 result rows, each carrying the parameters, ``mean_capacity``,
        ``rel_connected_time``, and ``split``.
    lambda_value
        The lambda this selection is for; one specialist per lambda.
    selection_split
        The split selection may look at. Rows from any other split are
        ignored rather than silently included -- selecting on a test
        split is the exact mistake this signature exists to prevent.
    """
    eligible = _eligible_rows(rows, split=selection_split, lambda_value=lambda_value)
    if not eligible:
        raise ValueError(
            f"no A3 rows from selection split {selection_split!r}; refusing to "
            "select on whatever else was passed in"
        )
    parameters, score = _best(_aggregate_by_parameters(eligible, lambda_value))
    return A3Selection(
        level=LEVEL_GLOBAL,
        lambda_value=lambda_value,
        parameters=parameters,
        selected_on=selection_split,
        score=score,
    )


def speed_bin_label(speed_kph: float, bins: Sequence[tuple[float, float]]) -> str | None:
    """Return the label of the bin containing ``speed_kph``.

    Bins are half-open ``[low, high)`` except the last, which includes
    its upper endpoint so the top of the declared range is covered.
    """
    for index, (low, high) in enumerate(bins):
        is_last = index == len(bins) - 1
        if low <= speed_kph < high or (is_last and speed_kph == high):
            return f"[{low:g},{high:g}{']' if is_last else ')'}"
    return None


def select_speed_aware(
    rows: Sequence[Mapping[str, Any]],
    *,
    lambda_value: float,
    selection_split: str,
    bins: Sequence[tuple[float, float]] = DEFAULT_SPEED_BINS,
) -> list[A3Selection]:
    """Choose one parameter set per predefined ID speed bin.

    Rows must carry ``mean_speed_kph`` (the UE's own sampled mean speed).
    Rows whose speed falls outside every bin are skipped: for the ID
    splits that cannot happen by construction, and for anything else the
    caller should be using :func:`clip_to_highest_bin` instead of
    silently extending a bin.
    """
    eligible = _eligible_rows(rows, split=selection_split, lambda_value=lambda_value)
    if not eligible:
        raise ValueError(
            f"no A3 rows from selection split {selection_split!r}"
        )

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible:
        label = speed_bin_label(float(row["mean_speed_kph"]), bins)
        if label is not None:
            grouped.setdefault(label, []).append(row)

    missing = [
        f"[{low:g},{high:g}{']' if index == len(bins) - 1 else ')'}"
        for index, (low, high) in enumerate(bins)
    ]
    empty = [label for label in missing if label not in grouped]
    if empty:
        raise ValueError(
            f"speed bins with no results on {selection_split!r}: {empty}; "
            "a bin without data cannot be given a parameter set"
        )

    selections: list[A3Selection] = []
    for label, bin_rows in sorted(grouped.items()):
        parameters, score = _best(_aggregate_by_parameters(bin_rows, lambda_value))
        selections.append(
            A3Selection(
                level=LEVEL_SPEED_AWARE,
                lambda_value=lambda_value,
                parameters=parameters,
                selected_on=selection_split,
                score=score,
                bin_label=label,
            )
        )
    return selections


def clip_to_highest_bin(
    selections: Sequence[A3Selection], bins: Sequence[tuple[float, float]] = DEFAULT_SPEED_BINS
) -> A3Selection:
    """Return the top bin's selection, for speeds above the ID range.

    OOD-Speed exceeds every ID bin. Clipping onto the highest one -- as
    opposed to fitting a new bin for it -- is what keeps the speed-aware
    level honest: it never sees OOD data.
    """
    if not selections:
        raise ValueError("no speed-aware selections to clip onto")
    top_low, top_high = bins[-1]
    label = f"[{top_low:g},{top_high:g}]"
    for selection in selections:
        if selection.bin_label == label:
            return selection
    raise ValueError(
        f"no selection for the highest bin {label}; available: "
        f"{[s.bin_label for s in selections]}"
    )


def select_split_oracle(
    rows: Sequence[Mapping[str, Any]], *, lambda_value: float, split: str
) -> A3Selection:
    """Re-optimize on the evaluation split itself.

    Explicitly marked non-deployable: it uses the test split's own
    results, so it is an upper bound on A3, never a baseline a real
    system could achieve.
    """
    eligible = _eligible_rows(rows, split=split, lambda_value=lambda_value)
    if not eligible:
        raise ValueError(f"no A3 rows from split {split!r}")
    parameters, score = _best(_aggregate_by_parameters(eligible, lambda_value))
    return A3Selection(
        level=LEVEL_SPLIT_ORACLE,
        lambda_value=lambda_value,
        parameters=parameters,
        selected_on=split,
        score=score,
        deployable=False,
    )


def build_selection_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    lambda_values: Sequence[float],
    selection_split: str,
    evaluation_splits: Sequence[str],
    bins: Sequence[tuple[float, float]] = DEFAULT_SPEED_BINS,
) -> list[dict[str, Any]]:
    """Build every A3 selection the release reports, as flat rows.

    Returns
    -------
    list of dict
        One row per (level, lambda, and where applicable bin/split),
        each carrying ``deployable`` and ``selected_on`` so a downstream
        reader cannot mistake the oracle for a baseline.
    """
    table: list[dict[str, Any]] = []
    for lambda_value in lambda_values:
        table.append(
            select_global(
                rows, lambda_value=lambda_value, selection_split=selection_split
            ).as_row()
        )
        table.extend(
            selection.as_row()
            for selection in select_speed_aware(
                rows,
                lambda_value=lambda_value,
                selection_split=selection_split,
                bins=bins,
            )
        )
        for split in evaluation_splits:
            table.append(
                {
                    **select_split_oracle(
                        rows, lambda_value=lambda_value, split=split
                    ).as_row(),
                    "evaluated_on": split,
                }
            )
    return table


__all__ = [
    "A3Parameters",
    "A3Selection",
    "DEFAULT_SPEED_BINS",
    "LEVELS",
    "LEVEL_GLOBAL",
    "LEVEL_SPEED_AWARE",
    "LEVEL_SPLIT_ORACLE",
    "build_selection_table",
    "clip_to_highest_bin",
    "objective_score",
    "select_global",
    "select_speed_aware",
    "select_split_oracle",
    "speed_bin_label",
]
