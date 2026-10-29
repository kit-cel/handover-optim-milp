"""Select A3 profiles at global, speed-aware, and split-oracle levels.

The first two use the declared selection split. The split oracle uses
evaluation data and is not deployable. Scores equal mean capacity minus
lambda times outage fraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import A3_HYS_COLUMN, A3_OFFSET_COLUMN, A3_TTT_COLUMN

LEVEL_GLOBAL = "global"
LEVEL_SPEED_AWARE = "speed_aware"
LEVEL_SPLIT_ORACLE = "split_oracle"
LEVELS = (LEVEL_GLOBAL, LEVEL_SPEED_AWARE, LEVEL_SPLIT_ORACLE)

# Fixed speed bins; half-open except the last.
DEFAULT_SPEED_BINS: tuple[tuple[float, float], ...] = (
    (30.0, 50.0),
    (50.0, 70.0),
    (70.0, 90.0),
)


@dataclass(frozen=True, slots=True, order=True)
class A3Parameters:
    """A3 parameters; ordering makes score ties deterministic."""

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
    """Average each profile score across eligible trajectories."""
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
    """Return the highest score, breaking ties by parameters."""
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
    """Select one profile using only the declared selection split."""
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
    """Select one profile per populated speed bin on the selection split."""
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
    """Use the highest trained speed bin for faster UEs."""
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
    """Select on the evaluation split; this result is not deployable."""
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
    """Return global, speed-aware, and split-oracle selections."""
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
