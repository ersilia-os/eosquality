"""Compute reference-population diagnostics from self-kNN results."""

import numpy as np

from eosquality.reference.fit_state import ReferenceReport
from eosquality.utils.arrays import exclude_self_neighbors
from eosquality.utils.logging import logger
from eosquality.utils.stats import decay_score, robust_spread


def compute_reference_report(
    ref_repr: np.ndarray,
    knn_distances_with_self: np.ndarray,
    knn_indices_with_self: np.ndarray,
    exclude_self: bool = True,
) -> tuple[ReferenceReport, np.ndarray, np.ndarray]:
    """Compute reference quality metrics from self-kNN results.

    Parameters
    ----------
    ref_repr:
        Scaled reference feature array (n_ref, n_features).
    knn_distances_with_self:
        Neighbor distances. When ``exclude_self=True`` (default), expected
        shape is (n_ref, k+1) with the identity neighbor at column 0.
        When ``exclude_self=False``, expected shape is (n_ref, k) — already
        clean (used when FP self-kNN is pre-computed without self).
    knn_indices_with_self:
        Corresponding neighbor indices (same shape rules as above).
    exclude_self:
        If ``True`` (default), strip column 0 as the self-neighbor.
        If ``False``, use the arrays as-is.

    Returns
    -------
    report:
        ReferenceReport with summary quality metrics.
    knn_distances:
        Self-kNN distances with the identity neighbor removed. (n_ref, k).
    knn_indices:
        Corresponding indices. (n_ref, k).
    """
    if exclude_self:
        knn_distances, knn_indices = exclude_self_neighbors(
            knn_distances_with_self, knn_indices_with_self
        )
    else:
        knn_distances = knn_distances_with_self
        knn_indices = knn_indices_with_self

    mean_k_distances = knn_distances.mean(axis=1)  # (n_ref,)
    median_k_distance = float(np.median(mean_k_distances))

    # Cohesion: how close each reference point is to its neighbors relative
    # to the overall spread of reference distances.
    spread = robust_spread(mean_k_distances)
    cohesion_score = float(np.clip(1.0 - np.median(mean_k_distances) / (spread + 1e-12), 0.0, 1.0))

    # Fragmentation: normalized std of per-point mean distances.
    fragmentation_score = float(
        np.std(mean_k_distances) / (np.mean(mean_k_distances) + 1e-12)
    )
    fragmentation_score = float(np.clip(fragmentation_score, 0.0, 1.0))

    # Reference quality: mean support score of the reference against itself.
    per_point_support = decay_score(mean_k_distances, scale=median_k_distance)
    reference_quality = float(np.mean(per_point_support))

    notes: list[str] = []
    if cohesion_score < 0.3:
        notes.append("Low cohesion: reference population may be fragmented.")
    if fragmentation_score > 0.5:
        notes.append("High fragmentation: reference population is heterogeneous.")

    logger.debug(
        f"Reference diagnostics | n={len(ref_repr):,} samples"
        f" | median_k_dist={median_k_distance:.4f}"
        f" | cohesion={cohesion_score:.4f}"
        f" | fragmentation={fragmentation_score:.4f}"
        f" | quality={reference_quality:.4f}"
    )
    for note in notes:
        logger.warning(note)

    report = ReferenceReport(
        reference_quality=reference_quality,
        cohesion_score=cohesion_score,
        fragmentation_score=fragmentation_score,
        median_k_distance=median_k_distance,
        notes=notes,
    )
    return report, knn_distances, knn_indices
