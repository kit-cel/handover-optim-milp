"""Data loader."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..reference.rrc_utils import l3_filtering
from .ma_filter import MovingAverageFilter

if TYPE_CHECKING:
    from ..reference.rrc_config import RRCConfig
    from ..result_manager.episode_result import EpisodeResult


def _as_steps_ue_bs(
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Normalize input to shape (n_steps, n_ue, n_bs).
    Accepts:
      - (n_steps, n_bs) -> (n_steps, 1, n_bs)
      - (n_steps, n_ue, n_bs) unchanged
    """
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected x.ndim in {{2,3}}, got {x.ndim} with shape {x.shape}.")


def preprocess_dataset(
    data: "EpisodeResult",
    rrc_config: "RRCConfig",
    use_l3_filtering: bool = True,
    ue_no: int | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.int_],
    NDArray[np.int_],
]:
    """
    Preprocess dataset for multiple UEs.

    Returns
    -------
    rsrp_dbm : array, shape (n_bs, n_ue, n_steps)
    sinr_db  : array, shape (n_bs, n_ue, n_steps)
    q_in_mat : int array, shape (n_bs, n_ue, n_steps)
    q_out_mat: int array, shape (n_bs, n_ue, n_steps)

    Notes
    -----
    - L1 Qin/Qout moving-average filtering is applied per UE (independent filter state per UE).
    - L3 filtering is applied per UE and BS (recursive IIR along time).
    """
    raw_rsrp_dbm = _as_steps_ue_bs(np.asarray(data.rsrp))
    raw_sinr_db = _as_steps_ue_bs(np.asarray(data.sinr))

    n_steps, n_ue, n_bs = raw_rsrp_dbm.shape
    if ue_no is not None and (ue_no < 0 or ue_no >= n_ue):
        raise ValueError(f"ue_no must be in [0, {n_ue-1}], got {ue_no}.")

    if raw_sinr_db.shape != (n_steps, n_ue, n_bs):
        raise ValueError(
            "data.rsrp and data.sinr must have matching shapes after normalization. "
            f"Got rsrp {raw_rsrp_dbm.shape}, sinr {raw_sinr_db.shape}."
        )

    # --- L1 moving average filters for Qin/Qout (per UE) ---
    l1_dt_ms = int(rrc_config.t_res_ms)  # int(rrc_config.l1_filter_sample_interval_ms)
    qin_len_ms = int(rrc_config.l1_q_in_filter_len_ms)
    qout_len_ms = int(rrc_config.l1_q_out_filter_len_ms)

    qin_len_samples = qin_len_ms // l1_dt_ms
    qout_len_samples = qout_len_ms // l1_dt_ms
    if qin_len_samples <= 0 or qout_len_samples <= 0:
        raise ValueError(
            "Computed L1 filter lengths in samples must be positive. "
            f"Got qin_len_samples={qin_len_samples}, qout_len_samples={qout_len_samples}."
        )

    l1_sinr_db_qin = np.zeros_like(raw_sinr_db, dtype=np.float32)
    l1_sinr_db_qout = np.zeros_like(raw_sinr_db, dtype=np.float32)

    for ue in range(n_ue):
        maf_qin = MovingAverageFilter(qin_len_samples, dtype=np.float32)
        maf_qout = MovingAverageFilter(qout_len_samples, dtype=np.float32)
        for t in range(n_steps):
            l1_sinr_db_qin[t, ue, :] = maf_qin.step(raw_sinr_db[t, ue, :])
            l1_sinr_db_qout[t, ue, :] = maf_qout.step(raw_sinr_db[t, ue, :])

    q_in_mat = (l1_sinr_db_qin >= rrc_config.q_in_db).astype(int)
    q_out_mat = (l1_sinr_db_qout <= rrc_config.q_out_db).astype(int)

    # Convert to (n_bs, n_ue, n_steps)
    q_in_mat = np.transpose(q_in_mat, (2, 1, 0))
    q_out_mat = np.transpose(q_out_mat, (2, 1, 0))

    # --- L3 filtered RSRP and SINR ---
    if use_l3_filtering:
        rsrp_f = np.zeros_like(raw_rsrp_dbm, dtype=np.float32)
        sinr_f = np.zeros_like(raw_sinr_db, dtype=np.float32)

        for ue in range(n_ue):
            f_rsrp_old = raw_rsrp_dbm[0, ue, :].astype(np.float32, copy=True)
            f_sinr_old = raw_sinr_db[0, ue, :].astype(np.float32, copy=True)

            for t in range(n_steps):
                f_rsrp_new = l3_filtering(
                    f_old=f_rsrp_old,
                    m_new=raw_rsrp_dbm[t, ue, :].astype(np.float32, copy=False),
                    k=rrc_config.l3_filter_coef,
                    t_sample=rrc_config.t_res_ms,
                    t_ref=rrc_config.l3_filter_sample_reference_interval_ms,
                )
                f_sinr_new = l3_filtering(
                    f_old=f_sinr_old,
                    m_new=raw_sinr_db[t, ue, :].astype(np.float32, copy=False),
                    k=rrc_config.l3_filter_coef,
                    t_sample=rrc_config.t_res_ms,
                    t_ref=rrc_config.l3_filter_sample_reference_interval_ms,
                )

                rsrp_f[t, ue, :] = f_rsrp_new
                sinr_f[t, ue, :] = f_sinr_new

                f_rsrp_old = f_rsrp_new.astype(np.float32, copy=True)
                f_sinr_old = f_sinr_new.astype(np.float32, copy=True)

        rsrp_dbm = np.transpose(rsrp_f, (2, 1, 0))  # (n_bs, n_ue, n_steps)
        sinr_db = np.transpose(sinr_f, (2, 1, 0))  # (n_bs, n_ue, n_steps)
    else:
        rsrp_dbm = np.transpose(raw_rsrp_dbm, (2, 1, 0)).astype(np.float32, copy=False)
        sinr_db = np.transpose(raw_sinr_db, (2, 1, 0)).astype(np.float32, copy=False)

    if ue_no is not None:
        rsrp_dbm = rsrp_dbm[:, ue_no, :].squeeze()  # (n_bs, n_steps)
        sinr_db = sinr_db[:, ue_no, :].squeeze()  # (n_bs, n_steps)
        q_in_mat = q_in_mat[:, ue_no, :].squeeze()  # (n_bs, n_steps)
        q_out_mat = q_out_mat[:, ue_no, :].squeeze()  # (n_bs, n_steps)
    return rsrp_dbm, sinr_db, q_in_mat, q_out_mat
