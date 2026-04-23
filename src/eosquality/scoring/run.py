"""Compute per-sample scores for query data against a fitted reference."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.reference.fit_state import FitState
from eosquality.scoring.typicality import compute_typicality
from eosquality.utils.logging import logger
from eosquality.utils.stats import decay_score, geometric_mean


@dataclass
class RunResult:
    """Result returned by ErsiliaQuality.run().

    Attributes
    ----------
    scores:
        Per-query summary scores (one row per query sample).
    metadata:
        Global metadata about the run (reference size, k, reference_quality, ...).
    typicality_per_feature:
        ``(n_query, n_features)`` DataFrame of per-column typicality values —
        use this to drill into *which* column made a sample look atypical.
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]
    typicality_per_feature: pd.DataFrame = field(default_factory=pd.DataFrame)


def score_queries(
    query_repr: np.ndarray,
    query_raw: np.ndarray,
    query_knn_distances_raw: np.ndarray,
    query_knn_indices_raw: np.ndarray,
    fit_state: FitState,
    query_index: list[Any],
) -> RunResult:
    """Compute quality scores for each query sample.

    Parameters
    ----------
    query_repr:
        Scaled query feature array in [0, 1] (n_query, n_features). Drives
        the neighbor distance and consistency signals.
    query_raw:
        Raw query feature values (n_query, n_features). Drives typicality
        via the stored per-column empirical CDF.
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

    # Typicality: per-column tail probability under the reference CDF,
    # aggregated by geometric mean. Measures whether each individual feature
    # value is plausible on its own — complements support (which measures
    # whether the whole sample sits near a structural neighbor).
    scalers = fit_state.preprocess_state["scalers"]
    schema = fit_state.preprocess_state["schema"]
    quantile_levels = fit_state.preprocess_state.get("quantile_levels")
    typicality_per_feature, typicality_score = compute_typicality(
        raw_values=query_raw,
        scalers=scalers,
        column_names=list(schema.column_names),
        n_reference=n_ref,
        quantile_levels=quantile_levels,
    )

    quality_score = np.array(
        [
            geometric_mean(
                [
                    float(support_score[i]),
                    float(typicality_score[i]),
                    float(consistency_score[i]),
                ]
            )
            for i in range(n_query)
        ]
    )

    nearest_reference_ids = [
        [fit_state.reference_ids[j] for j in knn_indices[i]]
        for i in range(n_query)
    ]

    logger.debug(
        f"  support_score: mean={support_score.mean():.4f}"
        f" | typicality_score: mean={typicality_score.mean():.4f}"
        f" | consistency_score: mean={consistency_score.mean():.4f}"
        f" | quality_score: mean={quality_score.mean():.4f}"
    )

    scores = pd.DataFrame(
        {
            "quality_score": quality_score,
            "support_score": support_score,
            "typicality_score": typicality_score,
            "consistency_score": consistency_score,
            "distance_k_mean": distance_k_mean,
            "distance_k_max": distance_k_max,
            "nearest_reference_ids": nearest_reference_ids,
        },
        index=query_index,
    )

    per_feature = pd.DataFrame(
        typicality_per_feature,
        index=query_index,
        columns=list(schema.column_names),
    )

    metadata: dict[str, Any] = {
        "reference_quality": fit_state.reference_report.reference_quality,
        "reference_typicality": getattr(
            fit_state.reference_report, "reference_typicality", float("nan")
        ),
        "n_reference": len(fit_state.reference_ids),
        "k": k,
    }

    return RunResult(
        scores=scores,
        metadata=metadata,
        typicality_per_feature=per_feature,
    )
