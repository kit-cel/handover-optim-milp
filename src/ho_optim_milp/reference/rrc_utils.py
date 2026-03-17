"""Utility functions for RRC (Radio Resource Control) operations in 5G simulations."""

import numpy as np
from numpy.typing import NDArray


def l3_filtering(
    f_old: NDArray[np.floating],
    m_new: NDArray[np.floating],
    k: float,
    t_sample: int,
    t_ref: int = 200,
) -> NDArray[np.floating]:
    """
    L3 filtering with exponential smoothing.

    The formula used for the filtering is:
        F_n = (1 - alpha) * F_(n-1) + alpha * M_n
    where:
    - F_n is the new filtered value,
    - F_(n-1) is the previous filtered value,
    - M_n is the new measurement,
    - alpha is the filter weight calculated as:
        alpha = 2^(-k/4)

    Parameters
    ----------
    f_old :  NDArray[np.floating]
        The previous filtered RSRP/SINR values.
    m_new :  NDArray[np.floating]
        The new RSRP/SINR measurements.
    k : float
        The L3 filter coefficient.
    t_sample : int
        The sampling period of the measurements in milliseconds.
    t_ref : int, optional
        The reference period for the filtering in milliseconds. Default is 200 ms.

    Returns
    -------
    NDArray[np.floating]
        The filtered RSRP/SINR values after applying L3 filtering.

    Notes
    -----
    - The filter coefficient `k` should be in the range [0, 15].
    - Filtering is performed in the same domain as used for evaluation of reporting criteria,
        i.e., RSRP in dBm or RSRQ and SINR in dB.
    - If the sampling period `t_sample` is different from the reference period `t_ref`,
        the filter weight is adjusted accordingly to preeserve the same time characteristics.
    """
    a = 1 / (2 ** (k / 4))  # L3 filter weight
    a_new = 1 - (1 - a) ** (t_sample / t_ref)
    return (1 - a_new) * f_old + a_new * m_new
