"""L1 RLM filtering of precomputed radio traces.

Both the MILP and the reference simulations consume the *same* arrays, so
neither can gain an advantage from a preprocessing difference.

What this module does and, importantly, does not do
---------------------------------------------------
**L1 (radio link monitoring only).** A boxcar moving average of raw SINR
with a *separate* window per threshold (``l1_q_in_filter_len_ms`` vs.
``l1_q_out_filter_len_ms``), compared against ``q_in_db``/``q_out_db`` to
produce the RLM in-sync/out-of-sync matrices. Every method shares these,
so radio link monitoring is identical across MILP, A3, no-handover and
any learned policy.

**No L3 filtering.** L3 is a *measurement reporting* filter
(TS 38.331 clause 5.5.3.2) and belongs to the entity that evaluates
reporting criteria -- :class:`~mobile_network_sim.rrc.measurement_handler
.MeasurementHandler`, which applies it internally. Applying it here as
well filtered the A3 reference's decision input twice, which made A3
react measurably too late: on one 6-UE episode, 59 handovers and 8 RLFs
instead of 75 and 4.

The MILP has no measurement entity and needs no L3 filtering: its
Q-in/Q-out come from L1 above, and its objective is computed from raw
SINR.

**Raw SINR for capacity.** The achievable rate is a property of the
channel, not of a measurement report, so it is computed from the
unfiltered wideband SINR -- identically for every method, which is what
makes the comparison fair.

Filtering runs at ``t_res_ms`` (one measurement per RRC step);
``l1_filter_sample_interval_ms`` is unused because the simulator has no
faster PHY-update cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..reference.rrc_utils import l3_filtering
from .ma_filter import MovingAverageFilter

if TYPE_CHECKING:
    from ..reference.rrc_config import RRCConfig

    from ..result_manager.episode_result import EpisodeResult

# Filtering is carried out in float32, matching the dtype the released
# dataset was generated with. Promoting to float64 here would change the
# stored capacity values in the last digits and break exact reproduction.
_FILTER_DTYPE = np.float32


@dataclass(frozen=True, slots=True)
class PreprocessedTrace:
    """Filtered radio trace for one episode, shaped ``(n_bs, n_ue, n_steps)``.

    Attributes
    ----------
    rsrp_dbm
        Unfiltered RSRP. Fed to the RRC state machine, which applies L3
        filtering itself; feeding it a pre-filtered value would filter
        twice.
    sinr_db
        Unfiltered wideband SINR. The capacity/objective is computed from
        this, and it is likewise fed to the RRC unfiltered.
    q_in_mat, q_out_mat
        RLM in-sync / out-of-sync indicators from L1-filtered SINR.
    l1_sinr_qin_db, l1_sinr_qout_db
        The two continuous L1 streams before thresholding.  They differ
        whenever Qin and Qout use different window lengths.
    """

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
        """Return the single-UE slice, still shaped ``(n_bs, 1, n_steps)``.

        The UE axis is kept rather than squeezed so downstream shape
        checks stay uniform; use :meth:`as_2d` where a bare
        ``(n_bs, n_steps)`` array is wanted.
        """
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
    """Boxcar-average raw SINR over ``window_samples`` along the time axis.

    Vectorized over ``(n_ue, n_bs)``: :class:`BoxcarFilter` is elementwise,
    so feeding it the whole ``(n_ue, n_bs)`` frame per step is arithmetically
    identical to the historical per-UE loop while doing ``n_ue`` times fewer
    Python-level iterations.
    """
    filtered = np.zeros_like(raw_sinr_db, dtype=_FILTER_DTYPE)
    boxcar = MovingAverageFilter(window_samples, dtype=_FILTER_DTYPE)
    for t in range(raw_sinr_db.shape[0]):
        filtered[t] = boxcar.step(raw_sinr_db[t].astype(_FILTER_DTYPE, copy=False))
    return filtered


def preprocess_episode(
    episode: "EpisodeResult",
    rrc_config: "RRCConfig",
) -> PreprocessedTrace:
    """Filter one episode's raw radio trace for MILP/A3 consumption.

    Parameters
    ----------
    episode
        Episode container providing raw ``rsrp``/``sinr`` arrays shaped
        ``(n_steps, n_ue, n_bs)`` (or ``(n_steps, n_bs)`` for a single UE).
    rrc_config
        Supplies the L1 window lengths, the RLM thresholds, and
        ``t_res_ms``. Its L3 parameters are used by the RRC state
        machine, not here.

    Returns
    -------
    PreprocessedTrace
        Arrays shaped ``(n_bs, n_ue, n_steps)``.
    """
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


def preprocess_dataset(
    data: "EpisodeResult",
    rrc_config: "RRCConfig",
    use_l3_filtering: bool = True,
    ue_no: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare radio arrays for the standalone RRC reference.

    The MILP uses :func:`preprocess_episode` directly. The standalone RRC
    implementation expects L3-filtered measurement inputs, whereas the
    simulator RRC implementation applies L3 internally.
    """
    trace = preprocess_episode(data, rrc_config)
    rsrp = trace.rsrp_dbm
    sinr = trace.sinr_db
    if use_l3_filtering:
        rsrp = _l3_measurements(rsrp, rrc_config)
        sinr = _l3_measurements(sinr, rrc_config)
    if ue_no is not None:
        return (
            rsrp[:, ue_no, :],
            sinr[:, ue_no, :],
            trace.q_in_mat[:, ue_no, :],
            trace.q_out_mat[:, ue_no, :],
        )
    return rsrp, sinr, trace.q_in_mat, trace.q_out_mat


def _l3_measurements(values: np.ndarray, config: "RRCConfig") -> np.ndarray:
    """Apply the standalone reference's per-cell L3 measurement filter."""
    n_bs, n_ue, n_steps = values.shape
    filtered = np.empty((n_bs, n_ue, n_steps), dtype=_FILTER_DTYPE)
    for ue_id in range(n_ue):
        previous = values[:, ue_id, 0].astype(_FILTER_DTYPE, copy=True)
        for step in range(n_steps):
            current = l3_filtering(
                f_old=previous,
                m_new=values[:, ue_id, step].astype(_FILTER_DTYPE, copy=False),
                k=config.l3_filter_coef,
                t_sample=config.t_res_ms,
                t_ref=config.l3_filter_sample_reference_interval_ms,
            )
            filtered[:, ue_id, step] = current
            previous = current.astype(_FILTER_DTYPE, copy=True)
    return filtered
