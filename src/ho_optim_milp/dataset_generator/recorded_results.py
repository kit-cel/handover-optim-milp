"""Canonical results computed from one stored network episode."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from gurobipy import GRB
from mobile_network_sim.rrc import RRCConfig, RRCStepUE
from mobile_network_sim.rrc.flags import RRCFlags

from ..common.metrics import LINK_FAILURE_CAUSES, build_metric_row, count_ping_pongs
from ..common.trajectory import milp_step_frame, reference_step_frame
from ..dataloader.preprocessing import preprocess_episode
from ..optimization.milp import HandoverOptimizerMILP
from ..optimization.optim_config import OptimConfig
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



def _reference_frames(
    episode: EpisodeResult, config: RRCConfig, trigger_handovers: bool
) -> list[pd.DataFrame]:
    trace = preprocess_episode(episode, config)
    n_cells, n_ue, n_steps = trace.sinr_db.shape
    ues = [RRCStepUE(ue_id=ue_id, rrc_config=config) for ue_id in range(n_ue)]
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
                rsrp_dbm=trace.rsrp_dbm[:, ue_id, step],
                sinr_db=trace.sinr_db[:, ue_id, step],
                q_in=trace.q_in_mat[:, ue_id, step],
                q_out=trace.q_out_mat[:, ue_id, step], action=None,
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
            flags, trace.sinr_db, ue_id,
            snapshots=[row[ue_id] for row in snapshots],
            **{name: np.asarray(values)[:, ue_id, :]
               for name, values in measurements.items()},
        )
        capacity = np.log2(1.0 + 10.0 ** (trace.sinr_db[:, ue_id, :] / 10.0))
        for cell in range(n_cells):
            frame[f"capacity_{cell}"] = capacity[cell].astype(np.float32)
        frames.append(frame)
    return frames


def reference_recorded_episode(
    episode: EpisodeResult,
    config: RRCConfig,
    *,
    lambda_r: float,
    trigger_handovers: bool = True,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Replay A3 or no-HO and return one canonical result per UE."""
    frames = _reference_frames(episode, config, trigger_handovers)
    return [(_reference_metrics(frame, config, lambda_r, ue_id), frame)
            for ue_id, frame in enumerate(frames)]
