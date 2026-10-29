"""Loading one episode out of a merged network-simulation HDF5 file.

Reads only the requested episode's arrays rather than the whole file: a
split's H5 holds every episode, and a job array of thousands of tasks
each loading all of them would dominate the runtime and the I/O budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..result_manager.episode_result import EpisodeResult

_ARRAY_FIELDS = (
    "ue_pos",
    "distances_2d",
    "distances_3d",
    "rsrp",
    "sinr",
    "ue_move_angle",
)


def episode_group_names(path: str | Path) -> list[str]:
    """Return the episode group names of a network H5 file, in order."""
    with h5py.File(path, "r") as handle:
        return sorted(name for name in handle if name.startswith("ep_"))


def episode_shapes(path: str | Path) -> list[tuple[str, int, int, int]]:
    """Return ``(group_name, n_steps, n_ue, n_bs)`` for every episode.

    Used by manifest generation to learn how many UEs each episode has
    without materializing any radio data.
    """
    shapes: list[tuple[str, int, int, int]] = []
    with h5py.File(path, "r") as handle:
        for name in sorted(n for n in handle if n.startswith("ep_")):
            sinr = handle[f"{name}/sinr"]
            if sinr.ndim == 3:
                n_steps, n_ue, n_bs = sinr.shape
            else:
                n_steps, n_bs = sinr.shape
                n_ue = 1
            shapes.append((name, int(n_steps), int(n_ue), int(n_bs)))
    return shapes


def load_episode(
    path: str | Path, episode_idx: int, *, max_steps: int | None = None
) -> EpisodeResult:
    """Load one recorded episode, optionally limited to a step prefix."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        names = sorted(name for name in handle if name.startswith("ep_"))
        if not 0 <= episode_idx < len(names):
            raise IndexError(
                f"{path} has {len(names)} episodes; episode_idx={episode_idx} "
                "is out of range"
            )
        name = names[episode_idx]
        group = handle[name]

        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        arrays: dict[str, np.ndarray] = {}
        for field in _ARRAY_FIELDS:
            if field in group:
                selection = slice(None, max_steps) if max_steps is not None else slice(None)
                arrays[field] = np.asarray(group[field][selection], dtype=np.float32)

        meta = _read_group_json(group.get("meta"))
        config_common = _read_group_json(handle.get("meta"))

    return EpisodeResult.from_dict(
        ep_name=name,
        data=arrays,
        meta=meta,
        config_common=config_common.get("config_common", config_common),
        strict=False,
        warn=False,
    )


def _read_group_json(group: Any) -> dict[str, Any]:
    """Decode JSON-valued datasets of an H5 metadata group."""
    if group is None:
        return {}
    payload: dict[str, Any] = {}
    for key in group:
        raw = group[key][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                payload[key] = json.loads(raw)
            except json.JSONDecodeError:
                payload[key] = raw
        else:
            payload[key] = raw
    return payload
