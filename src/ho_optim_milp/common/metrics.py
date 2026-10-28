"""Metrics shared by offline MILP and reference trajectories."""

import numpy as np
from numpy.typing import NDArray


def count_ping_pongs(
    pcell: NDArray[np.integer],
    ho_completed: NDArray[np.integer] | NDArray[np.bool_],
    t_mts_steps: int,
) -> int:
    """Count A-B-A handovers completed within the inclusive time window."""
    cells = np.asarray(pcell)
    events = np.asarray(ho_completed)
    if cells.ndim != 1 or events.ndim != 1 or cells.shape != events.shape:
        raise ValueError("pcell and ho_completed must be matching 1-D arrays")
    if t_mts_steps < 0:
        raise ValueError("t_mts_steps must be non-negative")
    completions = np.flatnonzero(events)
    return sum(
        int(first > 0 and second - first <= t_mts_steps and cells[first - 1] == cells[second])
        for first, second in zip(completions[:-1], completions[1:])
    )
