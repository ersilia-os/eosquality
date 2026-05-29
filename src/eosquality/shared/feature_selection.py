"""Pick a non-redundant subset of features by correlation-cluster medoid.

Given an eosframes-scaled reference matrix, compute the absolute Pearson
correlation between columns, treat ``1 - |corr|`` as a distance, run
hierarchical clustering with ``max_features`` clusters, and keep the
medoid of each cluster. The result is a list of column names of length
at most ``max_features``, preserving the original column order.

This is a fit-time helper. Downstream code is expected to slice the
scaled reference array to these columns via
``SharedFitState.filter_features``; the original :class:`Schema` is left
untouched so the eosframes scaler still operates on the full input.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from eosquality.utils.logging import logger


def select_features_by_correlation(
    ref_scaled: np.ndarray,
    column_names: list[str],
    max_features: int | None,
) -> list[str]:
    """Return a non-redundant subset of column names, at most ``max_features`` long.

    Parameters
    ----------
    ref_scaled:
        ``(n_ref, n_features)`` eosframes-scaled reference array.
    column_names:
        Column names matching ``ref_scaled.shape[1]``.
    max_features:
        Cap on the returned subset. ``None`` disables selection.

    Returns
    -------
    list[str]
        Selected column names, in the order they appear in ``column_names``.
        If selection is skipped (cap is ``None`` or already satisfied), the
        full list of column names is returned.
    """
    n_features = len(column_names)
    if ref_scaled.ndim != 2 or ref_scaled.shape[1] != n_features:
        raise ValueError(
            f"ref_scaled shape {ref_scaled.shape} does not match "
            f"len(column_names)={n_features}."
        )
    if max_features is None or n_features <= max_features:
        logger.debug(
            f"feature selection: skipped (n_features={n_features}, "
            f"max_features={max_features})"
        )
        return list(column_names)
    if max_features <= 0:
        raise ValueError(f"max_features must be positive or None; got {max_features}.")

    distances = _correlation_distance_matrix(ref_scaled)
    # Hierarchical clustering on the condensed upper triangle.
    condensed = squareform(distances, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=max_features, criterion="maxclust")

    selected_indices: list[int] = []
    for cid in np.unique(cluster_ids):
        members = np.where(cluster_ids == cid)[0]
        if members.size == 1:
            selected_indices.append(int(members[0]))
            continue
        # Medoid = column with smallest mean intra-cluster distance.
        sub = distances[np.ix_(members, members)]
        medoid_local = int(np.argmin(sub.mean(axis=1)))
        selected_indices.append(int(members[medoid_local]))

    selected_indices.sort()
    selected = [column_names[i] for i in selected_indices]
    logger.debug(
        f"feature selection: {n_features} → {len(selected)} "
        f"(max_features={max_features})"
    )
    return selected


def _correlation_distance_matrix(ref_scaled: np.ndarray) -> np.ndarray:
    """Symmetric ``(n_features, n_features)`` matrix of ``1 - |corr|`` values.

    NaN-safe via :func:`numpy.ma.corrcoef`. Columns with zero variance
    (after scaling) have undefined correlation; we map any non-finite
    entries to distance 1.0 so they cluster with nothing and survive as
    independent features (which matches the "preserve diverse signals"
    spirit of the selection).
    """
    masked = np.ma.masked_invalid(ref_scaled)
    corr = np.ma.corrcoef(masked, rowvar=False)
    corr = np.asarray(corr.filled(0.0), dtype=np.float64)
    distances = 1.0 - np.abs(corr)
    distances[~np.isfinite(distances)] = 1.0
    np.fill_diagonal(distances, 0.0)
    # Symmetrize against tiny floating-point drift so squareform accepts it.
    distances = 0.5 * (distances + distances.T)
    np.clip(distances, 0.0, 1.0, out=distances)
    return distances
