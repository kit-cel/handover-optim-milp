"""Optional adapter for the separately distributed mobile network simulator."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from ..common.metrics import count_ping_pongs
from ..dataloader.preprocessing import preprocess_dataset
from ..reference.rrc_config import RRCConfig
from ..result_manager.episode_result import EpisodeResult


def _simulator_types() -> tuple[Any, Any, Any]:
    try:
        from mobile_network_sim.rrc import RRCConfig as SimulatorRRCConfig, RRCStepUE
        from mobile_network_sim.runners import RadioRRCSimulationConfig, RadioRRCSimulationRunner
    except ImportError as error:
        raise RuntimeError(
            "This operation requires mobile-network-sim; install ho_optim_milp[simulator]."
        ) from error
    return SimulatorRRCConfig, RRCStepUE, (RadioRRCSimulationConfig, RadioRRCSimulationRunner)


def _simulator_rrc_config(config: RRCConfig) -> Any:
    simulator_rrc, _, _ = _simulator_types()
    old_fields = config.model_dump(mode="python")
    fields = {name: value for name, value in old_fields.items()
              if name in simulator_rrc.model_fields}
    fields["phy_update_interval_ms"] = config.t_res_ms
    fields["msr_interval_ms"] = config.t_res_ms
    return simulator_rrc.model_validate(fields)


def run_a3_reference(episode: EpisodeResult, config: RRCConfig) -> list[dict[str, Any]]:
    """Replay stored measurements through the simulator's RRC state machine."""
    _, rrc_step_ue, _ = _simulator_types()
    sim_config = _simulator_rrc_config(config)
    rsrp, sinr, q_in, q_out = preprocess_dataset(
        episode, config, use_l3_filtering=False
    )
    n_cells, n_ues, n_steps = sinr.shape
    ues = [rrc_step_ue(ue_id=index, rrc_config=sim_config) for index in range(n_ues)]
    for ue in ues:
        ue.set_learning_mode(enable_agent_mode=False, enable_eval=True)
    pcell = np.empty((n_steps, n_ues), dtype=np.int32)
    ho_end = np.zeros((n_steps, n_ues), dtype=np.int8)
    ho_running = np.zeros_like(ho_end)
    hof = np.zeros_like(ho_end)
    rlf_start = np.zeros_like(ho_end)
    rlf_running = np.zeros_like(ho_end)
    for step in range(n_steps):
        for index, ue in enumerate(ues):
            ue.step(
                time_ms=step * config.t_res_ms,
                rsrp_dbm=rsrp[:, index, step],
                sinr_db=sinr[:, index, step],
                q_in=q_in[:, index, step],
                q_out=q_out[:, index, step],
                action=None,
            )
            pcell[step, index] = ue.mob.get_pcell_if_available()
            ho_end[step, index] = int(ue.flags_now.ho_success)
            ho_running[step, index] = int(ue.timers.exec.active)
            hof[step, index] = int(ue.flags_now.ho_failure)
            rlf_start[step, index] = int(ue.flags_now.rlf_start)
            rlf_running[step, index] = int(ue.timers.rlf_recovery.active)

    capacity = np.log2(1.0 + 10.0 ** (sinr / 10.0))
    duration_s = n_steps * config.t_res_ms / 1000.0
    rows: list[dict[str, Any]] = []
    for index in range(n_ues):
        serving_cell = pcell[:, index]
        valid_cell = (serving_cell >= 0) & (serving_cell < n_cells)
        outage = ((ho_running[:, index] + rlf_running[:, index]) > 0) | ~valid_cell
        serving = capacity[np.clip(serving_cell, 0, n_cells - 1), index,
                           np.arange(n_steps)]
        delivered = np.where(outage, 0.0, serving)
        handovers = int(ho_end[:, index].sum())
        failures = int(hof[:, index].sum())
        ping_pongs = count_ping_pongs(
            pcell[:, index], ho_end[:, index], config.t_mts_ms // config.t_res_ms
        )
        rows.append({
            "ue_id": index,
            "simulated_time_s": duration_s,
            "simulated_steps": n_steps,
            "mean_capacity": float(delivered.mean()),
            "max_mean_capacity": float(np.max(capacity[:, index, :], axis=0).mean()),
            "rel_connected_time": float((~outage).mean()),
            "num_ho": handovers,
            "ho_per_s": handovers / duration_s,
            "num_pp": ping_pongs,
            "pp_per_s": ping_pongs / duration_s,
            "pp_rate": ping_pongs / handovers if handovers else float("nan"),
            "num_rlf": int(rlf_start[:, index].sum()),
            "num_hof": failures,
            "hof_rate": failures / (failures + handovers) if failures + handovers else 0.0,
        })
    return rows


def episode_from_steps(
    steps: Sequence[Any], *, ep_name: str = "ep_0000", config: Any | None = None
) -> EpisodeResult:
    """Convert a completed simulator trajectory into the independent input type."""
    _simulator_types()
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
    """Generate and save one recorded episode using the optional simulator."""
    _, _, (config_type, runner_type) = _simulator_types()
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = config_type.model_validate(yaml.safe_load(handle))
    runner = runner_type(config)
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
                                                          "producer": "mobile-network-sim"}))
        group = handle.create_group(episode.ep_name)
        for name, values in episode.to_dict().items():
            if name != "ep_name":
                group.create_dataset(name, data=values)
    temporary.replace(path)
    return path
