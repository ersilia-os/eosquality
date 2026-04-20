"""Statistical utility functions used across the package."""

import numpy as np


def geometric_mean(values: list[float]) -> float:
    """Geometric mean of a list of positive values."""
    arr = np.array(values, dtype=float)
    arr = np.clip(arr, 1e-12, None)
    return float(np.exp(np.mean(np.log(arr))))


def robust_spread(x: np.ndarray) -> float:
    """Interquartile range as a robust measure of spread.

    Returns 1.0 if IQR is zero to avoid division by zero downstream.
    """
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    return iqr if iqr > 0 else 1.0


def decay_score(distance: np.ndarray, scale: float) -> np.ndarray:
    """Map distances to (0, 1] via exponential decay: exp(-distance / scale)."""
    if scale <= 0:
        scale = 1.0
    return np.exp(-distance / scale)
