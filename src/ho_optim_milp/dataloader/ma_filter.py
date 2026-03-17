"""
Moving average filter.
"""

import numpy as np
from numpy.typing import NDArray


class MovingAverageFilter:
    """Moving average filter."""

    def __init__(self, filter_len: int, dtype: type = np.float64) -> None:
        """
        Initialize the moving average filter.

        Parameters
        ----------
        filter_len : int
            Window length for moving average.
        """
        if filter_len <= 0:
            raise ValueError("Filter length must be positive integer.")
        self.filter_len = int(filter_len)
        self._cap = self.filter_len

        # Lazy buffer/sum allocation based on first input shape
        self._buf: NDArray[np.floating] | None = None
        self._sum: NDArray[np.floating] | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype = dtype

        self._head = 0
        self._total = 0

    @property
    def count(self) -> int:
        """Number of samples seen so far (capped by window size for mean)."""
        return min(self._total, self.filter_len)

    @property
    def full(self) -> bool:
        """Whether the window is fully populated."""
        return self._total >= self.filter_len

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Shape of the expected sample arrays, ``None`` until first sample."""
        return self._shape

    def reset(self) -> None:
        """Reset the filter state but keep allocated buffers (fast)."""
        self._buf = None
        self._sum = None
        self._shape = None

        self._head = 0
        self._total = 0

    def get_mean(self) -> NDArray[np.floating] | None:
        """Return the current mean without pushing a new sample.
        Returns ``None`` if no samples were observed yet.
        """
        if self._sum is None or self.count == 0:
            return None
        return self._sum / self.count

    def step(
        self, values: np.floating | NDArray[np.floating]
    ) -> np.floating | NDArray[np.floating]:
        """Push a new sample and return the moving average.

        Parameters
        ----------
        values : NDArray[np.floating]
            New sample array of arbitrary shape.

        Returns
        -------
        NDArray[np.floating]
            Mean over the last ``filter_len`` samples.
        """
        values = np.asarray(values, dtype=self._dtype)
        if self._buf is None:
            self._lazy_init(values)

        if self._buf is None or self._sum is None:
            raise ValueError("Buffer not initialized")

        buf = self._buf
        sum_values = self._sum

        if self.full:
            idx_old = (self._head - self.filter_len) % self._cap
            sum_values -= buf[idx_old]

        buf[self._head] = values
        sum_values += values

        self._head = (self._head + 1) % self._cap
        self._total += 1

        return sum_values / self.count

    def _lazy_init(self, values: NDArray[np.floating]) -> None:
        self._buf = np.zeros((self._cap, *values.shape), dtype=self._dtype)
        self._sum = np.zeros_like(values, dtype=self._dtype)
        self._shape = values.shape
