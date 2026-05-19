"""Fit the shared kNN state from the vector index's precomputed self-kNN."""

from __future__ import annotations

import numpy as np

from eosquality.knn.state import KnnFitState
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex


def fit_knn(
    shared: SharedFitState,
    ref_repr: np.ndarray,
    vector_index: VectorIndex,
    k: int,
) -> KnnFitState:
    """Read the reference's FP self-kNN from the precomputed index arrays.

    No neighbor search runs here. The FP-selected indices and their
    Tanimoto distances were materialized once when the vector index was
    built (with the identity neighbor already stripped); this function
    just slices them at ``k`` and reduces the distances to a per-row
    mean so Support can build its CDF.

    Output-space distances are *not* computed here — that work lives in
    :meth:`Consistency.fit`, where it is actually consumed.

    Parameters
    ----------
    shared:
        Pre-fit shared state (only used for logging context).
    ref_repr:
        Scaled reference array from :func:`fit_shared` (``(n_ref, n_features)``).
        Persisted on the returned state for Consistency / run-time use.
    vector_index:
        Loaded VectorIndex aligned row-for-row with ``ref_repr``. The
        VectorIndex is *not* persisted on the returned state — only
        ``k`` is. At load time the index is re-resolved via the
        canonical library resolver using ``shared.metadata.library_id``.
    k:
        Number of neighbors.
    """
    knn_indices = vector_index.self_knn_indices(k)
    fp_distances = vector_index.self_knn_distances(k).astype(np.float64)
    mean_fp_distances = fp_distances.mean(axis=1)

    logger.debug(
        f"FP self-kNN loaded | k={k} | n_ref={knn_indices.shape[0]:,}"
        f" | median mean_fp_dist={float(np.median(mean_fp_distances)):.4f}"
    )

    return KnnFitState(
        ref_repr=ref_repr,
        k=k,
        mean_fp_distances=mean_fp_distances,
        reference_knn_indices=knn_indices,
    )
