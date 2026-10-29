"""Portable storage helpers for datasets built from recorded radio data."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mobile_network_sim.rrc import RRCConfig

from ..optimization.optim_config import OptimConfig


ENVIRONMENT_FIELDS = {"log_dir", "simulation_id", "path_to_config"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def scientific_config(config: OptimConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json", exclude=ENVIRONMENT_FIELDS)
    payload["rrc"] = config.rrc.model_dump(mode="json", exclude=ENVIRONMENT_FIELDS)
    solver = payload.get("gurobi")
    if isinstance(solver, dict):
        solver.pop("parameter_file", None)
        solver.pop("gurobi_log_file", None)
    return payload


def validate_episode(episode: Any, rrc: RRCConfig) -> tuple[int, int]:
    rsrp = np.asarray(episode.rsrp)
    sinr = np.asarray(episode.sinr)
    if rsrp.shape != sinr.shape or sinr.ndim not in (2, 3) or sinr.size == 0:
        raise ValueError("Episode RSRP and SINR must be nonempty matching 2-D or 3-D arrays")
    if not np.isfinite(rsrp).all() or not np.isfinite(sinr).all():
        raise ValueError("Episode RSRP and SINR must contain only finite values")
    stored = episode.config.get("t_res_ms")
    if stored is None and isinstance(episode.config.get("rrc"), dict):
        stored = episode.config["rrc"].get("t_res_ms")
    if stored is not None and int(stored) != rrc.t_res_ms:
        raise ValueError(
            f"RRC time resolution {rrc.t_res_ms} ms differs from stored {stored} ms"
        )
    return int(sinr.shape[0]), int(sinr.shape[1]) if sinr.ndim == 3 else 1
