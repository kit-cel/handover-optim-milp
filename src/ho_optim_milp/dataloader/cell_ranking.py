"""Deterministic cell ranking for optimizer constraints."""

import numpy as np
from numpy.typing import NDArray


def rank1_neighbor_table(rsrp_dbm: NDArray[np.floating]) -> NDArray[np.int32]:
    """Return the strongest other cell for every source cell and time step."""
    values = np.asarray(rsrp_dbm)
    if values.ndim != 2:
        raise ValueError(f"rsrp_dbm must have shape (n_cells, n_steps); got {values.shape}")
    n_cells, n_steps = values.shape
    if n_cells == 0:
        raise ValueError("rsrp_dbm must contain at least one cell")
    if n_cells == 1:
        return np.full((1, n_steps), -1, dtype=np.int32)
    order = np.argsort(-values, axis=0, kind="stable")
    best, second = order[:2]
    targets = np.broadcast_to(best, (n_cells, n_steps)).astype(np.int32, copy=True)
    targets[best, np.arange(n_steps)] = second.astype(np.int32, copy=False)
    return targets
