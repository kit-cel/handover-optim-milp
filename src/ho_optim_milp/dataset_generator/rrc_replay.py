"""Replay stored MILP handovers through the simulator's RRC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mobile_network_sim.rrc import RRCStepUE
from mobile_network_sim.rrc.flags import RRCFlags

from ..common.schema import MILP_PCELL_COLUMN
from ..common.trajectory import reference_step_frame
from ..dataloader.preprocessing import preprocess_episode

if TYPE_CHECKING:
    from mobile_network_sim.rrc import RRCConfig

    from ..result_manager.episode_result import EpisodeResult


def _stored_pcell_column(frame: pd.DataFrame) -> str:
    if MILP_PCELL_COLUMN not in frame:
        raise ValueError("stored trajectory has no PCell column")
    return MILP_PCELL_COLUMN


def _completion_schedule(frame: pd.DataFrame, config: "RRCConfig") -> dict[int, int]:
    if "i_ho_exec_end" not in frame:
        raise ValueError("MILP trajectory has no i_ho_exec_end column")
    durations = (config.t_ho_prep_ms_simulated, config.t_ho_exec_ms_simulated)
    if any(duration % config.t_res_ms for duration in durations):
        raise ValueError("handover durations must be multiples of t_res_ms")
    lead = sum(durations) // config.t_res_ms
    pcell = frame[_stored_pcell_column(frame)].to_numpy(dtype=int)
    completions = np.flatnonzero(frame["i_ho_exec_end"].to_numpy(dtype=bool))
    schedule: dict[int, int] = {}
    for completion in completions:
        prep_start = int(completion) - lead
        if prep_start < 0:
            raise ValueError(f"completion at step {completion} is left-censored")
        target = int(pcell[completion])
        if prep_start in schedule and schedule[prep_start] != target:
            raise ValueError(f"conflicting MILP targets at step {prep_start}")
        schedule[prep_start] = target
    return schedule


def replay_stored_rrc_trajectory(
    config: "RRCConfig",
    episode: "EpisodeResult",
    *,
    ue_id: int,
    stored_steps: pd.DataFrame,
    method: str,
    validate: bool = True,
) -> pd.DataFrame:
    """Replay a stored MILP or no-HO trajectory and verify its step states."""
    if method not in ("milp", "no_ho"):
        raise ValueError("method must be 'milp' or 'no_ho'")
    stored = stored_steps.sort_values("time_step").reset_index(drop=True)
    trace = preprocess_episode(episode, config)
    if len(stored) != trace.n_steps:
        raise ValueError(
            f"stored/replay length mismatch: {len(stored)} != {trace.n_steps}"
        )
    if not 0 <= ue_id < trace.n_ue:
        raise ValueError(f"ue_id must be in [0, {trace.n_ue - 1}]")

    schedule = _completion_schedule(stored, config) if method == "milp" else {}
    pcell_column = _stored_pcell_column(stored)
    initial_pcell = int(stored[pcell_column].iloc[0])
    ue = RRCStepUE(ue_id=ue_id, rrc_config=config)
    ue.set_learning_mode(enable_agent_mode=True, enable_eval=True)

    flag_rows: dict[str, list[int]] = {
        name: []
        for name in (
            "pcell",
            "ho_exec_end",
            "ho_exec_running",
            "hof_detected",
            "rlf_start",
            "rlf_running",
            *RRCFlags.field_names(),
        )
    }
    snapshots: list[dict[str, Any]] = []
    measurements: dict[str, list[np.ndarray]] = {
        "l3_rsrp_dbm": [],
        "l3_sinr_db": [],
        "l1_qin_sinr_db": [],
        "l1_qout_sinr_db": [],
        "q_in": [],
        "q_out": [],
    }

    for step in range(trace.n_steps):
        action = schedule.get(step)
        accepted = ue.step(
            time_ms=step * config.t_res_ms,
            rsrp_dbm=trace.rsrp_dbm[:, ue_id, step],
            sinr_db=trace.sinr_db[:, ue_id, step],
            q_in=trace.q_in_mat[:, ue_id, step],
            q_out=trace.q_out_mat[:, ue_id, step],
            action=0 if action is None else action + 1,
            action_mode="target_cell",
            initial_pcell=initial_pcell if step == 0 else None,
        )
        if action is not None and not accepted:
            raise RuntimeError(
                f"stored MILP action to cell {action} rejected at step {step}"
            )
        flag_rows["pcell"].append(int(ue.mob.get_pcell_if_available()))
        flag_rows["ho_exec_end"].append(int(ue.flags_now.ho_success))
        flag_rows["ho_exec_running"].append(int(ue.timers.exec.active))
        flag_rows["hof_detected"].append(int(ue.flags_now.ho_failure))
        flag_rows["rlf_start"].append(int(ue.flags_now.rlf_start))
        flag_rows["rlf_running"].append(int(ue.timers.rlf_recovery.active))
        for name in RRCFlags.field_names():
            if name == "rlf_start":
                continue
            flag_rows[name].append(int(getattr(ue.flags_now, name)))
        snapshots.append(ue.snapshot())
        measurements["l3_rsrp_dbm"].append(np.asarray(ue.msr_handler.rsrp_dbm).copy())
        measurements["l3_sinr_db"].append(np.asarray(ue.msr_handler.sinr_db).copy())
        measurements["l1_qin_sinr_db"].append(
            np.asarray(ue.msr_handler.l1_sinr_qin_db).copy()
        )
        measurements["l1_qout_sinr_db"].append(
            np.asarray(ue.msr_handler.l1_sinr_qout_db).copy()
        )
        measurements["q_in"].append(np.asarray(ue.msr_handler.q_in).copy())
        measurements["q_out"].append(np.asarray(ue.msr_handler.q_out).copy())

    flags = {name: np.asarray(values)[:, None] for name, values in flag_rows.items()}
    frame = reference_step_frame(
        flags,
        trace.sinr_db[:, ue_id : ue_id + 1, :],
        0,
        snapshots=snapshots,
        **{name: np.asarray(values) for name, values in measurements.items()},
    )
    if validate:
        expected_pcell = stored[pcell_column].to_numpy(dtype=int)
        actual_pcell = frame["pcell"].to_numpy(dtype=int)
        if not np.array_equal(expected_pcell, actual_pcell):
            mismatch = int(np.flatnonzero(expected_pcell != actual_pcell)[0])
            raise RuntimeError(f"RRC replay PCell mismatch at step {mismatch}")
        for stored_name, replay_name in (
            ("i_ho_exec_end", "i_ho_exec_end"),
            ("i_ho_exec_running", "i_ho_exec_running"),
            ("i_rlf_start", "i_rlf_start"),
            ("i_rlf_running", "i_rlf_running"),
        ):
            if stored_name not in stored:
                continue
            expected = stored[stored_name].to_numpy(dtype=bool)
            actual = frame[replay_name].to_numpy(dtype=bool)
            if not np.array_equal(expected, actual):
                mismatch = int(np.flatnonzero(expected != actual)[0])
                raise RuntimeError(
                    f"RRC replay {stored_name} mismatch at step {mismatch}"
                )
    return frame


__all__ = ["replay_stored_rrc_trajectory"]
