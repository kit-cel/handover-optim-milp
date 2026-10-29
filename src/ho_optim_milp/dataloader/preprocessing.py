"""Prepare shared raw radio and L1 RLM inputs for MILP and A3.

L3 filtering belongs to the RRC measurement handler; capacity uses raw SINR.
L1 filtering uses float32 to match saved dataset values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mobile_network_sim.rrc.rrc_utils import BoxcarFilter

if TYPE_CHECKING:
    from mobile_network_sim.rrc import RRCConfig

    from ..result_manager.episode_result import EpisodeResult

# Keep float32 for parity with stored dataset values.
_FILTER_DTYPE = np.float32


@dataclass(frozen=True, slots=True)
class PreprocessedTrace:
    """Raw radio data and L1 RLM indicators, shaped (cells, UEs, steps)."""

    rsrp_dbm: NDArray[np.floating]
    sinr_db: NDArray[np.floating]
    q_in_mat: NDArray[np.integer]
    q_out_mat: NDArray[np.integer]
    l1_sinr_qin_db: NDArray[np.floating]
    l1_sinr_qout_db: NDArray[np.floating]

    @property
    def n_bs(self) -> int:
        """Number of candidate cells."""
        return int(self.sinr_db.shape[0])

    @property
    def n_ue(self) -> int:
        """Number of UEs in the episode."""
        return int(self.sinr_db.shape[1])

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.sinr_db.shape[2])

    def for_ue(self, ue_idx: int) -> "PreprocessedTrace":
        """Keep the UE axis when selecting one UE."""
        if not 0 <= ue_idx < self.n_ue:
            raise ValueError(f"ue_idx must be in [0, {self.n_ue - 1}], got {ue_idx}")
        sl = slice(ue_idx, ue_idx + 1)
        return PreprocessedTrace(
            rsrp_dbm=self.rsrp_dbm[:, sl, :],
            sinr_db=self.sinr_db[:, sl, :],
            q_in_mat=self.q_in_mat[:, sl, :],
            q_out_mat=self.q_out_mat[:, sl, :],
            l1_sinr_qin_db=self.l1_sinr_qin_db[:, sl, :],
            l1_sinr_qout_db=self.l1_sinr_qout_db[:, sl, :],
        )

    def as_2d(
        self, ue_idx: int
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.integer],
        NDArray[np.integer],
    ]:
        """Return ``(rsrp, sinr, q_in, q_out)`` for one UE as ``(n_bs, n_steps)``."""
        if not 0 <= ue_idx < self.n_ue:
            raise ValueError(f"ue_idx must be in [0, {self.n_ue - 1}], got {ue_idx}")
        return (
            self.rsrp_dbm[:, ue_idx, :],
            self.sinr_db[:, ue_idx, :],
            self.q_in_mat[:, ue_idx, :],
            self.q_out_mat[:, ue_idx, :],
        )


def _as_steps_ue_bs(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Normalize a raw trace to ``(n_steps, n_ue, n_bs)``."""
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected x.ndim in {{2, 3}}, got {x.ndim} with shape {x.shape}.")


def _l1_indicators(
    raw_sinr_db: NDArray[np.floating],
    *,
    window_samples: int,
) -> NDArray[np.floating]:
    """Apply the L1 boxcar filter along the time axis."""
    filtered = np.zeros_like(raw_sinr_db, dtype=_FILTER_DTYPE)
    boxcar = BoxcarFilter(window_samples)
    for t in range(raw_sinr_db.shape[0]):
        filtered[t] = boxcar.step(raw_sinr_db[t].astype(_FILTER_DTYPE, copy=False))
    return filtered


def preprocess_episode(
    episode: "EpisodeResult",
    rrc_config: "RRCConfig",
) -> PreprocessedTrace:
    """Produce raw radio arrays and L1 RLM indicators for MILP and A3."""
    raw_rsrp_dbm = _as_steps_ue_bs(np.asarray(episode.rsrp))
    raw_sinr_db = _as_steps_ue_bs(np.asarray(episode.sinr))

    if raw_sinr_db.shape != raw_rsrp_dbm.shape:
        raise ValueError(
            "episode.rsrp and episode.sinr must have matching shapes after "
            f"normalization; got rsrp {raw_rsrp_dbm.shape}, sinr {raw_sinr_db.shape}."
        )

    t_res_ms = int(rrc_config.t_res_ms)
    q_in_samples = int(rrc_config.l1_q_in_filter_len_ms) // t_res_ms
    q_out_samples = int(rrc_config.l1_q_out_filter_len_ms) // t_res_ms
    if q_in_samples <= 0 or q_out_samples <= 0:
        raise ValueError(
            "L1 filter lengths must span at least one sample; got "
            f"q_in={q_in_samples}, q_out={q_out_samples} samples at "
            f"t_res_ms={t_res_ms}."
        )

    l1_sinr_q_in = _l1_indicators(raw_sinr_db, window_samples=q_in_samples)
    l1_sinr_q_out = _l1_indicators(raw_sinr_db, window_samples=q_out_samples)
    q_in_mat = (l1_sinr_q_in >= rrc_config.q_in_db).astype(int)
    q_out_mat = (l1_sinr_q_out <= rrc_config.q_out_db).astype(int)

    rsrp = raw_rsrp_dbm.astype(_FILTER_DTYPE, copy=False)
    sinr = raw_sinr_db.astype(_FILTER_DTYPE, copy=False)

    # (n_steps, n_ue, n_bs) -> (n_bs, n_ue, n_steps)
    return PreprocessedTrace(
        rsrp_dbm=np.transpose(rsrp, (2, 1, 0)),
        sinr_db=np.transpose(sinr, (2, 1, 0)),
        q_in_mat=np.transpose(q_in_mat, (2, 1, 0)),
        q_out_mat=np.transpose(q_out_mat, (2, 1, 0)),
        l1_sinr_qin_db=np.transpose(l1_sinr_q_in, (2, 1, 0)),
        l1_sinr_qout_db=np.transpose(l1_sinr_q_out, (2, 1, 0)),
    )
