"""sklearn-based nearest-neighbor index."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SklearnNeighborIndex:
    """Wraps sklearn NearestNeighbors with a unified fit/query interface."""

    def __init__(self, k: int = 20, algorithm: str = "auto") -> None:
        self.k = k
        self.algorithm = algorithm
        self._nn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> "SklearnNeighborIndex":
        """Index the reference array X."""
        # Request k+1 so self-queries can strip the identity neighbor.
        self._nn = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm=self.algorithm,
            metric="euclidean",
        )
        self._nn.fit(X)
        return self

    def query(self, X: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) for the k nearest neighbors of each row in X.

        Returns k+1 neighbors so callers can exclude the query point itself
        when the query set is the same as the reference.
        """
        if self._nn is None:
            raise RuntimeError("SklearnNeighborIndex must be fitted before query().")
        n_neighbors = (k or self.k) + 1
        # Clamp to available points
        n_neighbors = min(n_neighbors, self._nn.n_samples_fit_)
        distances, indices = self._nn.kneighbors(X, n_neighbors=n_neighbors)
        return distances, indices
