"""Per-column typicality: 'does this value look like it came from the reference?'

For each query sample and each feature column, we compute a typicality in
[0, 1] from the stored per-column artifact:

- **continuous / count / proportion**: tail probability under the reference
  empirical CDF. ``cdf(x)`` is linearly interpolated on the stored quantile
  grid; typicality = ``2 * min(cdf, 1 - cdf)``, so a value at the reference
  median scores 1.0 and a value at an extreme scores near 0.
- **binary**: ``min(1.0, 2 * class_freq[v])``. A balanced binary (p0=p1=0.5)
  scores 1.0 on both classes; a 90/10 imbalance scores 1.0 for the majority
  and 0.2 for the minority.
- **constant columns**: 1.0 regardless of query value (no information).

The aggregate per query is a geometric mean across features, with an eps floor
tied to the reference size to prevent a single extreme feature collapsing the
score to zero. The floor is ``1 / (2 * n_reference)`` — the Laplace-smoothed
tail probability of an observation more extreme than any reference point.
"""

from __future__ import annotations

import numpy as np

# Shared quantile grid from the preprocessing pipeline. Stored once per
# pipeline state, so at call time we read it from the state dict.
_DEFAULT_QUANTILE_LEVELS = np.linspace(0.0, 1.0, 101)


def compute_typicality(
    raw_values: np.ndarray,
    scalers: dict,
    column_names: list[str],
    n_reference: int,
    quantile_levels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature and aggregate typicality for a batch of queries.

    Parameters
    ----------
    raw_values:
        ``(n_query, n_features)`` float array of RAW query values (not the
        [0, 1] distance representation). Column order must match
        ``column_names``.
    scalers:
        Per-column dict produced by :mod:`eosquality.preprocess.pipeline`.
        Each entry must contain ``kind``, ``is_constant``, ``quantiles``
        (for continuous/count/proportion), and ``class_freq`` (for binary).
    column_names:
        Names of the feature columns in order.
    n_reference:
        Number of reference samples, used to compute the eps floor.
    quantile_levels:
        The quantile grid used during fit. If ``None``, falls back to the
        current default (101 uniform levels in [0, 1]).

    Returns
    -------
    per_feature:
        ``(n_query, n_features)`` typicality scores in ``[eps, 1]``.
    aggregate:
        ``(n_query,)`` aggregated typicality in ``[eps, 1]`` — the geometric
        mean across features.
    """
    if quantile_levels is None:
        quantile_levels = _DEFAULT_QUANTILE_LEVELS

    n_query = raw_values.shape[0]
    n_features = raw_values.shape[1]
    if n_features == 0:
        # No feature columns → typicality is vacuously 1.0.
        return np.ones((n_query, 0)), np.ones(n_query)

    # Floor so the geomean cannot collapse to zero when a single feature
    # value sits beyond the most extreme reference observation.
    eps = 1.0 / (2.0 * max(n_reference, 1))

    per_feature = np.empty((n_query, n_features), dtype=np.float64)
    for j, col in enumerate(column_names):
        params = scalers[col]
        per_feature[:, j] = _typicality_for_column(
            raw_values[:, j],
            params,
            quantile_levels,
            eps,
        )

    # Geometric mean across features: exp(mean(log(factors))).
    # All factors are >= eps > 0 by construction, so the log is safe.
    aggregate = np.exp(np.log(per_feature).mean(axis=1))
    return per_feature, aggregate


def _typicality_for_column(
    values: np.ndarray,
    params: dict,
    quantile_levels: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Per-column typicality scores for a 1-D array of query values."""
    kind = params.get("kind", "continuous")

    if params.get("is_constant", False):
        return np.ones_like(values, dtype=np.float64)

    if kind == "binary":
        return _binary_typicality(values, params["class_freq"], eps)

    # continuous / count / proportion: interpolate CDF on the quantile grid
    quantiles = params.get("quantiles")
    if quantiles is None:
        # Legacy scaler without a quantile grid; treat as constant (no info).
        return np.ones_like(values, dtype=np.float64)
    return _cdf_typicality(values, quantiles, quantile_levels, params["anchor"], eps)


def _binary_typicality(
    values: np.ndarray,
    class_freq: dict[float, float],
    eps: float,
) -> np.ndarray:
    """Typicality for binary columns.

    Returns ``min(1, 2 * class_freq[v])`` per value, floored at ``eps``.
    Unknown values (not in ``{0, 1}``) are mapped to the lower of the two
    class frequencies (floored at eps) — treated as maximally atypical
    since they do not fit the observed binary pattern.
    """
    p0 = float(class_freq.get(0.0, 0.0))
    p1 = float(class_freq.get(1.0, 0.0))
    typ_0 = min(1.0, 2.0 * p0) if p0 > 0 else eps
    typ_1 = min(1.0, 2.0 * p1) if p1 > 0 else eps
    unknown = max(eps, min(typ_0, typ_1))
    out = np.full_like(values, unknown, dtype=np.float64)
    # NaN handling: NaNs at this point mean missing input; treat as unknown.
    valid = ~np.isnan(values)
    out[valid & (values == 0.0)] = typ_0
    out[valid & (values == 1.0)] = typ_1
    return np.clip(out, eps, 1.0)


def _cdf_typicality(
    values: np.ndarray,
    quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    anchor_quantile: float,
    eps: float,
) -> np.ndarray:
    """Typicality via linear interpolation on a stored empirical CDF.

    ``quantiles[i]`` is the value at reference quantile ``quantile_levels[i]``.
    For a query value ``x`` we compute ``cdf(x)`` by interpolating
    ``quantile_levels`` as a function of ``quantiles`` — i.e. the inverse of
    the quantile function. Then tail prob = ``min(cdf, 1 - cdf)`` and
    typicality = ``2 * tail_prob``.

    NaN inputs map to the reference median (cdf = 0.5), which is maximally
    typical — NaN has no information and should not drag the score down.
    """
    # np.interp expects monotonic increasing xp. The stored quantiles are
    # already sorted (quantile output is non-decreasing). Ties produce flat
    # spans which interp handles correctly by returning the mapped value at
    # either endpoint of the span — acceptable for our use.
    filled = np.where(np.isnan(values), quantiles[len(quantiles) // 2], values)
    cdf = np.interp(filled, quantiles, quantile_levels, left=0.0, right=1.0)
    tail_prob = np.minimum(cdf, 1.0 - cdf)
    typ = 2.0 * tail_prob
    return np.clip(typ, eps, 1.0)
