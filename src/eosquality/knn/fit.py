"""Fit the shared kNN state from the vector index's precomputed self-kNN."""

from __future__ import annotations

import time

import numpy as np

from eosquality.knn.state import KnnFitState
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex


def fit_knn(
    shared: SharedFitState,
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
        Pre-fit shared state (only used for logging context). Consumers
        that need the scaled reference matrix read ``shared.ref_repr``
        directly.
    vector_index:
        Loaded VectorIndex aligned row-for-row with the reference. The
        VectorIndex is *not* persisted on the returned state — only
        ``k`` is. At load time the index is re-resolved via the
        canonical library resolver using ``shared.metadata.library_id``.
    k:
        Number of neighbors.
    """
    t0 = time.perf_counter()
    logger.info(f"fit_knn | loading FP self-kNN | k={k}")
    knn_indices = vector_index.self_knn_indices(k)
    fp_distances = vector_index.self_knn_distances(k).astype(np.float64)
    mean_fp_distances = fp_distances.mean(axis=1)
    logger.info(
        f"fit_knn | done | n_ref={knn_indices.shape[0]:,} k={k} | "
        f"{time.perf_counter() - t0:.1f}s"
    )

    return KnnFitState(
        k=k,
        mean_fp_distances=mean_fp_distances,
        reference_knn_indices=knn_indices,
    )
