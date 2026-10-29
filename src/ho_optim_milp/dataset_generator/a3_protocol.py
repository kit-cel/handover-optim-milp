"""Predeclared A3 search protocol for recorded network trajectories.

The coarse lattice and deterministic refinement rule can be recorded in
dataset metadata. Refinement is performed independently for each lambda on
the configured selection split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .a3_selection import A3Parameters, objective_score


@dataclass(frozen=True, slots=True)
class A3SearchProtocol:
    """Predeclared coarse-to-fine A3 parameter search."""

    protocol_id: str = "a3_coarse_to_fine_v1"
    coarse_ttt_ms: tuple[float, ...] = (0.0, 80.0, 160.0, 320.0, 640.0, 1280.0)
    coarse_hys_db: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0)
    coarse_offset_db: tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0)
    fine_ttt_ms: tuple[float, ...] = (
        0.0,
        40.0,
        80.0,
        160.0,
        240.0,
        320.0,
        480.0,
        640.0,
        960.0,
        1280.0,
    )
    fine_hys_db: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    fine_offset_db: tuple[float, ...] = (
        -8.0,
        -6.0,
        -4.0,
        -2.0,
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
    )
    top_k: int = 3
    ttt_neighbor_radius: int = 1
    hys_neighbor_radius: int = 1
    offset_neighbor_radius: int = 1
    exclude_coarse_from_refinement: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        for name in (
            "coarse_ttt_ms",
            "coarse_hys_db",
            "coarse_offset_db",
            "fine_ttt_ms",
            "fine_hys_db",
            "fine_offset_db",
        ):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and duplicate-free")

    def coarse_grid(self) -> tuple[tuple[float, float, float], ...]:
        """Return the full first-stage parameter lattice."""
        return tuple(
            (ttt, hys, offset)
            for ttt in self.coarse_ttt_ms
            for hys in self.coarse_hys_db
            for offset in self.coarse_offset_db
        )

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable protocol definition stored with every release."""
        payload = asdict(self)
        payload["coarse_grid_size"] = len(self.coarse_grid())
        return payload

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 over the complete search definition."""
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def refinement_grids(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        lambda_values: Sequence[float],
        selection_split: str,
    ) -> dict[float, tuple[tuple[float, float, float], ...]]:
        """Build deterministic fine grids around the top coarse candidates.

        Ranking uses only ``selection_split`` and the same objective as the
        MILP.  Ties are broken by the parameter tuple, so result-file order
        cannot change the refinement manifest.
        """
        result: dict[float, tuple[tuple[float, float, float], ...]] = {}
        for lambda_value in lambda_values:
            scores: dict[A3Parameters, list[float]] = {}
            for row in rows:
                if row.get("method") != "a3" or row.get("split") != selection_split:
                    continue
                if float(row.get("lambda", float("nan"))) != float(lambda_value):
                    continue
                parameters = A3Parameters(
                    float(row["ttt_ms"]),
                    float(row["hys_db"]),
                    float(row["offset_db"]),
                )
                scores.setdefault(parameters, []).append(
                    objective_score(
                        float(row["mean_capacity"]),
                        float(row["rel_connected_time"]),
                        float(lambda_value),
                    )
                )
            if not scores:
                raise ValueError(
                    f"no A3 rows for lambda={lambda_value:g} on {selection_split!r}"
                )
            ranked = sorted(
                ((sum(values) / len(values), params) for params, values in scores.items()),
                key=lambda item: (-item[0], item[1]),
            )
            candidates: set[tuple[float, float, float]] = set()
            for _score, params in ranked[: self.top_k]:
                ttts = _neighbors(self.fine_ttt_ms, params.ttt_ms, self.ttt_neighbor_radius)
                hyses = _neighbors(self.fine_hys_db, params.hys_db, self.hys_neighbor_radius)
                offsets = _neighbors(
                    self.fine_offset_db, params.offset_db, self.offset_neighbor_radius
                )
                candidates.update(
                    (ttt, hys, offset)
                    for ttt in ttts
                    for hys in hyses
                    for offset in offsets
                )
            if self.exclude_coarse_from_refinement:
                candidates.difference_update(self.coarse_grid())
            if not candidates:
                raise ValueError(
                    f"refinement grid for lambda={lambda_value:g} contains no new points"
                )
            result[float(lambda_value)] = tuple(sorted(candidates))
        return result


def _neighbors(
    lattice: Sequence[float], value: float, radius: int
) -> tuple[float, ...]:
    """Nearest lattice point plus ``radius`` neighbours on either side."""
    if radius < 0:
        raise ValueError("neighbor radius must be non-negative")
    nearest = min(range(len(lattice)), key=lambda index: abs(lattice[index] - value))
    start = max(0, nearest - radius)
    stop = min(len(lattice), nearest + radius + 1)
    return tuple(float(item) for item in lattice[start:stop])


DEFAULT_A3_PROTOCOL = A3SearchProtocol()


__all__ = ["A3SearchProtocol", "DEFAULT_A3_PROTOCOL"]
