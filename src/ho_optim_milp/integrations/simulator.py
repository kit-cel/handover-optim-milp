"""Generate stored radio trajectories with the released simulator."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import h5py
import mobile_network_sim
import numpy as np
import yaml
from mobile_network_sim.runners import RadioRRCSimulationConfig, RadioRRCSimulationRunner

from ..result_manager.episode_result import EpisodeResult





def episode_from_steps(
    steps: Sequence[Any], *, ep_name: str = "ep_0000", config: Any | None = None
) -> EpisodeResult:
    """Convert a completed simulator trajectory into the independent input type."""
    if not steps:
        raise ValueError("A complete nonempty simulator trajectory is required")
    frames = [step.radio.to_dict() for step in steps]
    keys = ("ue_pos", "distances_2d", "distances_3d", "rsrp", "sinr", "ue_move_angle")
    arrays = {name: np.asarray([frame[name] for frame in frames], dtype=np.float32)
              for name in keys}
    common = config.model_dump(mode="json") if config is not None else {}
    return EpisodeResult.from_dict(
        ep_name, arrays, config_common=common, strict=True, warn=False
    )


def generate_network_data(config_path: str | Path, output_path: str | Path) -> Path:
    """Generate and save one recorded radio episode."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = RadioRRCSimulationConfig.model_validate(yaml.safe_load(handle))
    runner = RadioRRCSimulationRunner(config)
    try:
        steps = runner.run()
    finally:
        runner.close()
    episode = episode_from_steps(steps, config=config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("config_common", data=json.dumps(episode.config_common))
        meta.create_dataset("release", data=json.dumps({"schema_version": 1,
                                                          "producer": "mobile-network-sim",
                                                          "producer_version": mobile_network_sim.__version__}))
        group = handle.create_group(episode.ep_name)
        for name, values in episode.to_dict().items():
            if name != "ep_name":
                group.create_dataset(name, data=values)
    temporary.replace(path)
    return path
