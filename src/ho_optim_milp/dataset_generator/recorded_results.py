"""Canonical results computed from one stored network episode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from gurobipy import GRB

from ..common.metrics import LINK_FAILURE_CAUSES, build_metric_row, count_ping_pongs
from ..common.trajectory import milp_step_frame, reference_step_frame
from ..dataloader.ma_filter import MovingAverageFilter
from ..dataloader.preprocessing import preprocess_dataset
from ..integrations.simulator import _simulator_rrc_config, _simulator_types
from ..optimization.milp import HandoverOptimizerMILP
from ..optimization.optim_config import OptimConfig
from ..reference.reference import RRCReferenceSimulation
from ..reference.rrc_config import RRCConfig
from ..result_manager.episode_result import EpisodeResult


def optimize_recorded_ue(
    episode: EpisodeResult, ue_id: int, config: OptimConfig
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Solve one UE and return canonical metrics and a flat step frame."""
    optimizer = HandoverOptimizerMILP(config)
    optimizer.load_data(episode, ue_id)
    optimizer.setup_model()
    optimizer.apply_warm_start_if_configured()
    optimizer.solve()
    if optimizer.model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"MILP did not prove optimality: {optimizer.model.Status}")
    frame = milp_step_frame(optimizer.extract_results())
    old = optimizer.get_aggregated_result_metrics()
    metrics = build_metric_row(
        ue_id=ue_id,
        simulated_time_s=old["simulated_time_s"],
        simulated_steps=old["simulated_steps"],
        mean_capacity=old["mean_capacity"],
        max_mean_capacity=old["max_mean_capacity"],
        rel_connected_time=old["rel_connected_time"],
        num_ho=old["num_ho"],
        num_pp=old["num_pp"],
        num_rlf=old["num_rlf"],
        link_failure_counts=None,
        objective_value=old["ho_objective_value"],
    )
    return metrics, frame


def _radio_arrays(episode: EpisodeResult) -> tuple[np.ndarray, np.ndarray]:
    raw_sinr = np.asarray(episode.sinr, dtype=np.float32)
    if raw_sinr.ndim == 2:
        raw_sinr = raw_sinr[:, None, :]
    return raw_sinr, np.transpose(raw_sinr, (2, 1, 0))


def _l1_measurements(raw_sinr: np.ndarray, config: RRCConfig) -> tuple[np.ndarray, np.ndarray]:
    steps, n_ue, n_cells = raw_sinr.shape
    results = []
    for window_ms in (config.l1_q_in_filter_len_ms, config.l1_q_out_filter_len_ms):
        length = window_ms // config.t_res_ms
        if length <= 0:
            raise ValueError("L1 filter window must contain at least one radio sample")
        values = np.empty((steps, n_ue, n_cells), dtype=np.float32)
        for ue_id in range(n_ue):
            filter_instance = MovingAverageFilter(length, dtype=np.float32)
            for step in range(steps):
                values[step, ue_id] = filter_instance.step(raw_sinr[step, ue_id])
        results.append(values)
    return results[0], results[1]


def _reference_metrics(
    frame: pd.DataFrame, config: RRCConfig, lambda_r: float, ue_id: int
) -> dict[str, Any]:
    steps = len(frame)
    causes = {cause: int(frame[f"i_{cause}"].sum()) for cause in LINK_FAILURE_CAUSES}
    pcell = frame["pcell"].to_numpy(dtype=np.int32)
    ho_end = frame["i_ho_exec_end"].to_numpy(dtype=np.int8)
    num_ho = int(ho_end.sum())
    num_pp = count_ping_pongs(pcell, ho_end, config.t_mts_ms // config.t_res_ms)
    connected = float(frame["i_connected"].mean())
    mean_capacity = float(frame["delivered_capacity"].mean())
    capacity_columns = [f"capacity_{cell}" for cell in range(
        len([name for name in frame if name.startswith("sinr_db_")])
    )]
    max_mean_capacity = float(frame[capacity_columns].max(axis=1).mean())
    return build_metric_row(
        ue_id=ue_id,
        simulated_time_s=steps * config.t_res_ms / 1000.0,
        simulated_steps=steps,
        mean_capacity=mean_capacity,
        max_mean_capacity=max_mean_capacity,
        rel_connected_time=connected,
        num_ho=num_ho,
        num_pp=num_pp,
        num_rlf=int(frame["i_rlf_start"].sum()),
        link_failure_counts=causes,
        objective_value=mean_capacity - lambda_r * (1.0 - connected),
    )


def _standalone_reference(
    episode: EpisodeResult, config: RRCConfig, trigger_handovers: bool
) -> list[pd.DataFrame]:
    reference = RRCReferenceSimulation(
        config, episode, trigger_handovers=trigger_handovers
    )
    reference.run()
    raw_sinr, sinr_by_cell = _radio_arrays(episode)
    l1_qin, l1_qout = _l1_measurements(raw_sinr, config)
    l3_rsrp, l3_sinr, q_in, q_out = preprocess_dataset(
        episode, config, use_l3_filtering=True
    )
    snapshots = reference.step_snapshots
    if len(snapshots) != raw_sinr.shape[0]:
        raise RuntimeError("Standalone RRC snapshots do not match radio samples")
    state_fields = {
        "pcell": "pcell", "ho_exec_end": "flag_ho_success",
        "ho_exec_running": "ho_exec_ongoing", "hof_detected": "flag_ho_failure",
        "rlf_start": "flag_rlf_start", "rlf_running": "rlf_timer_active",
    }
    flags = {
        name: np.asarray([[row[field] for row in sample] for sample in snapshots],
                         dtype=np.int32)
        for name, field in state_fields.items()
    }
    for name in reference.ue_list[0].get_flags():
        if name not in flags:
            flags[name] = np.asarray(
                [[row[f"flag_{name}"] for row in sample] for sample in snapshots],
                dtype=np.int32,
            )
    frames = []
    for ue_id in range(reference.n_ue):
        frame = reference_step_frame(
            flags, sinr_by_cell, ue_id,
            snapshots=[row[ue_id] for row in snapshots],
            l3_rsrp_dbm=l3_rsrp[:, ue_id, :].T,
            l3_sinr_db=l3_sinr[:, ue_id, :].T,
            l1_qin_sinr_db=l1_qin[:, ue_id, :],
            l1_qout_sinr_db=l1_qout[:, ue_id, :],
            q_in=q_in[:, ue_id, :].T,
            q_out=q_out[:, ue_id, :].T,
        )
        if "reporting_cell" in frame and "target_cell" in frame:
            frame["target_cell"] = frame["target_cell"].combine_first(
                frame["reporting_cell"]
            )
        capacity = np.log2(1.0 + 10.0 ** (raw_sinr[:, ue_id, :] / 10.0))
        for cell in range(capacity.shape[1]):
            frame[f"capacity_{cell}"] = capacity[:, cell].astype(np.float32)
        frames.append(frame)
    return frames


def _simulator_reference(
    episode: EpisodeResult, config: RRCConfig, trigger_handovers: bool
) -> list[pd.DataFrame]:
    from mobile_network_sim.rrc.flags import RRCFlags

    _, rrc_step_ue, _ = _simulator_types()
    sim_config = _simulator_rrc_config(config)
    raw_sinr, sinr_by_cell = _radio_arrays(episode)
    rsrp, sinr, q_in, q_out = preprocess_dataset(
        episode, config, use_l3_filtering=False
    )
    n_cells, n_ue, n_steps = sinr.shape
    ues = [rrc_step_ue(ue_id=ue_id, rrc_config=sim_config) for ue_id in range(n_ue)]
    for ue in ues:
        ue.set_learning_mode(enable_agent_mode=not trigger_handovers, enable_eval=True)
    names = ("pcell", "ho_exec_end", "ho_exec_running", "hof_detected",
             "rlf_start", "rlf_running", *RRCFlags.field_names())
    rows: dict[str, list[list[int]]] = {name: [] for name in names}
    snapshots: list[list[dict[str, Any]]] = []
    measurements: dict[str, list[list[np.ndarray]]] = {
        name: [] for name in ("l3_rsrp_dbm", "l3_sinr_db", "l1_qin_sinr_db",
                           "l1_qout_sinr_db", "q_in", "q_out")
    }
    for step in range(n_steps):
        for ue_id, ue in enumerate(ues):
            ue.step(
                time_ms=step * config.t_res_ms,
                rsrp_dbm=rsrp[:, ue_id, step], sinr_db=sinr[:, ue_id, step],
                q_in=q_in[:, ue_id, step], q_out=q_out[:, ue_id, step], action=None,
            )
        rows["pcell"].append([ue.mob.get_pcell_if_available() for ue in ues])
        rows["ho_exec_end"].append([int(ue.flags_now.ho_success) for ue in ues])
        rows["ho_exec_running"].append([int(ue.timers.exec.active) for ue in ues])
        rows["hof_detected"].append([int(ue.flags_now.ho_failure) for ue in ues])
        rows["rlf_start"].append([int(ue.flags_now.rlf_start) for ue in ues])
        rows["rlf_running"].append([int(ue.timers.rlf_recovery.active) for ue in ues])
        for name in RRCFlags.field_names():
            if name != "rlf_start":
                rows[name].append([int(getattr(ue.flags_now, name)) for ue in ues])
        snapshots.append([ue.snapshot() for ue in ues])
        for name in measurements:
            attribute = {
                "l3_rsrp_dbm": "rsrp_dbm", "l3_sinr_db": "sinr_db",
                "l1_qin_sinr_db": "l1_sinr_qin_db",
                "l1_qout_sinr_db": "l1_sinr_qout_db", "q_in": "q_in", "q_out": "q_out",
            }[name]
            measurements[name].append([
                np.asarray(getattr(ue.msr_handler, attribute)).copy() for ue in ues
            ])
    flags = {name: np.asarray(values) for name, values in rows.items()}
    frames = []
    for ue_id in range(n_ue):
        frame = reference_step_frame(
            flags, sinr_by_cell, ue_id,
            snapshots=[row[ue_id] for row in snapshots],
            **{name: np.asarray(values)[:, ue_id, :]
               for name, values in measurements.items()},
        )
        capacity = np.log2(1.0 + 10.0 ** (raw_sinr[:, ue_id, :] / 10.0))
        for cell in range(n_cells):
            frame[f"capacity_{cell}"] = capacity[:, cell].astype(np.float32)
        frames.append(frame)
    return frames


def reference_recorded_episode(
    episode: EpisodeResult,
    config: RRCConfig,
    *,
    lambda_r: float,
    backend: str = "standalone",
    trigger_handovers: bool = True,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Replay A3 or no-HO and return one canonical result per UE."""
    if backend == "standalone":
        frames = _standalone_reference(episode, config, trigger_handovers)
    elif backend == "simulator":
        frames = _simulator_reference(episode, config, trigger_handovers)
    else:
        raise ValueError(f"Unknown A3 backend: {backend}")
    return [(_reference_metrics(frame, config, lambda_r, ue_id), frame)
            for ue_id, frame in enumerate(frames)]
