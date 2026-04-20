"""Array utility functions used across the package."""

import numpy as np


def safe_nanmean(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Mean that returns 0.0 for all-NaN slices instead of raising."""
    with np.errstate(all="ignore"):
        result = np.nanmean(x, axis=axis)
    if np.ndim(result) == 0:
        return float(0.0 if np.isnan(result) else result)
    result = np.where(np.isnan(result), 0.0, result)
    return result


def bounded_clip(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clip array values to [lo, hi]."""
    return np.clip(x, lo, hi)


def exclude_self_neighbors(
    distances: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the first neighbor column (assumed to be the query point itself).

    Used when querying the reference index against itself.
    """
    return distances[:, 1:], indices[:, 1:]
