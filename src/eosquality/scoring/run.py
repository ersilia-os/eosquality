"""Compute per-sample scores for query data against a fitted reference."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eosquality.reference.fit_state import FitState
from eosquality.utils.arrays import exclude_self_neighbors
from eosquality.utils.logging import logger
from eosquality.utils.stats import decay_score, geometric_mean


@dataclass
class RunResult:
    """Result returned by ErsiliaQuality.run()."""

    scores: pd.DataFrame
    metadata: dict[str, Any]


def score_queries(
    query_repr: np.ndarray,
    query_knn_distances_raw: np.ndarray,
    query_knn_indices_raw: np.ndarray,
    fit_state: FitState,
    query_index: list[Any],
) -> RunResult:
    """Compute quality scores for each query sample.

    Parameters
    ----------
    query_repr:
        Scaled query feature array (n_query, n_features).
    query_knn_distances_raw:
        Neighbor distances returned by the index. Shape (n_query, k+1).
        The first column may or may not be self; we use k columns only.
    query_knn_indices_raw:
        Corresponding neighbor indices.
    fit_state:
        Fitted state from ErsiliaQuality.fit().
    query_index:
        Row index labels for the output DataFrame.
    """
    k = fit_state.config.neighbors.k
    median_k_dist = fit_state.reference_report.median_k_distance
    n_ref = len(fit_state.reference_ids)

    # Use only the first k columns (query points are not in the reference index,
    # so there is no identity neighbor to strip).
    knn_distances = query_knn_distances_raw[:, :k]
    knn_indices = query_knn_indices_raw[:, :k]

    n_query = len(query_repr)
    logger.debug(
        f"Scoring {n_query:,} queries | k={k} | n_ref={n_ref:,}"
        f" | median_ref_k_dist={median_k_dist:.4f}"
    )

    distance_k_mean = knn_distances.mean(axis=1)
    distance_k_max = knn_distances.max(axis=1)

    support_score = decay_score(distance_k_mean, scale=median_k_dist)
    consistency_score = decay_score(
        knn_distances.std(axis=1), scale=median_k_dist
    )

    quality_score = np.array(
        [
            geometric_mean([float(support_score[i]), float(consistency_score[i])])
            for i in range(n_query)
        ]
    )

    nearest_reference_ids = [
        [fit_state.reference_ids[j] for j in knn_indices[i]]
        for i in range(n_query)
    ]

    logger.debug(
        f"  support_score: mean={support_score.mean():.4f}"
        f" | consistency_score: mean={consistency_score.mean():.4f}"
        f" | quality_score: mean={quality_score.mean():.4f}"
    )

    # Intrinsic richness: mean per-column deviation from the reference baseline.
    # anchor_i is the normalized median of the reference for column i — the
    # "baseline" position in [0, 1]. A value at the baseline contributes 0;
    # a value far from it contributes up to 1.
    scalers = fit_state.preprocess_state["scalers"]
    schema = fit_state.preprocess_state["schema"]
    anchors = np.array([scalers[col]["anchor"] for col in schema.column_names])
    intrinsic_richness = np.abs(query_repr - anchors).mean(axis=1)
    logger.debug(f"  intrinsic_richness: mean={intrinsic_richness.mean():.4f}")

    scores = pd.DataFrame(
        {
            "quality_score": quality_score,
            "support_score": support_score,
            "consistency_score": consistency_score,
            "intrinsic_richness": intrinsic_richness,
            "distance_k_mean": distance_k_mean,
            "distance_k_max": distance_k_max,
            "effective_feature_fraction": np.ones(n_query),
            "nearest_reference_ids": nearest_reference_ids,
        },
        index=query_index,
    )

    metadata: dict[str, Any] = {
        "reference_quality": fit_state.reference_report.reference_quality,
        "n_reference": len(fit_state.reference_ids),
        "k": k,
    }

    return RunResult(scores=scores, metadata=metadata)
