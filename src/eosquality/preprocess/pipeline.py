"""Preprocessing pipeline: shape-aware, kind-aware normalization.

Per column we fit and persist two parallel artifacts:

1. **Scaling params** that drive the [0, 1] distance representation.
   - ``continuous`` / ``count``: percentile clip to [p1, p99] → [0, 1], with an
     optional log1p pre-transform when the data is sparse (>5% exact zeros)
     *and* heavy-tailed (non-zero range ratio > 10). The distribution-based
     log1p trigger naturally fires for most real counts.
   - ``binary`` / ``proportion``: pass-through — values are already in [0, 1].

2. **Quantile grid** (empirical CDF) that drives the typicality score.
   - ``continuous`` / ``count``: 101 quantiles on raw values (pre-log1p)
     interpolated linearly at query time.
   - ``binary``: ``class_freq = {0.0: p0, 1.0: p1}`` stored instead of a grid
     — typicality for a binary query value is ``min(1, 2 * class_freq[v])``.
   - ``proportion``: quantile grid on the raw [0, 1] values.
   - Constant columns are flagged ``is_constant=True``; their typicality
     contribution is 1.0 (no information either way).

Storing ``kind`` inside each scaler dict lets loaded pipelines round-trip
without consulting external metadata, and ``scaler_schema_version`` makes
future migrations detectable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eosquality.schema.models import Schema
from eosquality.utils.logging import logger

# Tolerance for treating a scale as zero (constant column).
_SCALE_EPS = 1e-9

# Current on-disk schema for the per-column scaler dict.
_SCALER_SCHEMA_VERSION = 2

# Fixed quantile grid shared across all columns. Stored once per pipeline so
# each column only needs its own quantile *values*.
_QUANTILE_LEVELS = np.linspace(0.0, 1.0, 101)


class PreprocessPipeline:
    """Fits and applies per-column kind-aware normalization.

    Parameters
    ----------
    schema:
        Column definitions that control which columns are processed.
    column_kinds:
        Mapping of column name → detected kind (``"binary"``, ``"count"``,
        ``"proportion"``, ``"continuous"``). Unknown or missing keys default
        to ``"continuous"``.
    """

    def __init__(
        self,
        schema: Schema,
        column_kinds: dict[str, str] | None = None,
        # strategy is accepted for backward-compat on load but ignored
        strategy: str = "type_aware",
    ) -> None:
        if strategy != "type_aware":
            raise ValueError(
                f"Preprocessing strategy '{strategy}' is no longer supported. "
                "The RobustScaler ('robust') has been replaced by type-aware "
                "percentile normalization. Please refit your model."
            )
        self._schema = schema
        self._column_kinds: dict[str, str] = column_kinds or {}
        self._scalers: dict[str, dict] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit + transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit normalizers on *df* and return the [0, 1] distance array."""
        n_cols = len(self._schema.column_names)
        logger.debug(f"Fitting normalization | {n_cols} column(s) | {len(df):,} samples")
        arrays = []
        for col in self._schema.column_names:
            kind = self._column_kinds.get(col, "continuous")
            values = df[col].to_numpy(dtype=float)
            params, transformed = _fit_column(values, kind=kind)
            self._scalers[col] = params
            logger.debug(
                f"  {col} [{kind}]: use_log1p={params['use_log1p']}"
                f" | p1={params['p1']:.4g} p99={params['p99']:.4g}"
                f" | anchor={params['anchor']:.3f}"
                f" | constant={params['is_constant']}"
            )
            arrays.append(transformed)
        self._is_fitted = True
        result = np.column_stack(arrays) if arrays else np.empty((len(df), 0))
        logger.debug(f"Normalization complete | output shape={result.shape}")
        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted normalizers to *df*; returns the [0, 1] distance array."""
        if not self._is_fitted:
            raise RuntimeError(
                "PreprocessPipeline must be fitted before transform()."
            )
        arrays = []
        for col in self._schema.column_names:
            values = df[col].to_numpy(dtype=float)
            arrays.append(_apply_column(values, self._scalers[col]))
        return np.column_stack(arrays) if arrays else np.empty((len(df), 0))

    def raw_numeric_values(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw float values for the fitted columns, shape (n, n_features).

        Typicality is computed on the *raw* scale (pre-log1p, pre-clip) because
        the stored quantile grid lives on raw values — this preserves shape
        information that percentile clipping would flatten.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "PreprocessPipeline must be fitted before raw_numeric_values()."
            )
        if not self._schema.column_names:
            return np.empty((len(df), 0))
        return df[list(self._schema.column_names)].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Serialization state
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a serialisable state dict for persistence."""
        return {
            "schema": self._schema,
            "scalers": self._scalers,
            "strategy": "type_aware",
            "scaler_schema_version": _SCALER_SCHEMA_VERSION,
            "quantile_levels": _QUANTILE_LEVELS,
        }

    @classmethod
    def from_state(cls, state: dict) -> "PreprocessPipeline":
        """Reconstruct a fitted pipeline from a persisted state dict."""
        strategy = state.get("strategy", "type_aware")
        pipeline = cls(schema=state["schema"], strategy=strategy)
        pipeline._scalers = state["scalers"]
        pipeline._is_fitted = True
        return pipeline


# ------------------------------------------------------------------
# Kind dispatcher
# ------------------------------------------------------------------


def _fit_column(values: np.ndarray, kind: str) -> tuple[dict, np.ndarray]:
    """Fit per-column params and return (params, distance_repr).

    Dispatches by ``kind``. The returned distance_repr is always in [0, 1]
    with the same length as ``values``; NaNs are mapped to the column's
    anchor (the center of the distribution) since they have no other
    principled fallback in a distance computation.
    """
    if kind == "binary":
        return _fit_binary(values)
    if kind == "proportion":
        return _fit_proportion(values)
    # count and continuous share the distribution-aware percentile path;
    # a column declared "count" does not short-circuit the trigger because
    # many count columns are tight ranges that don't need log1p.
    return _fit_type_aware(values, kind=kind)


def _apply_column(values: np.ndarray, params: dict) -> np.ndarray:
    """Apply pre-fitted kind-aware normalization to *values*."""
    kind = params.get("kind", "continuous")
    if kind == "binary":
        return _apply_binary(values, params)
    if kind == "proportion":
        return _apply_proportion(values, params)
    return _apply_type_aware(values, params)


# ------------------------------------------------------------------
# Continuous / count: percentile + optional log1p (current behavior)
# ------------------------------------------------------------------


def _fit_type_aware(values: np.ndarray, kind: str = "continuous") -> tuple[dict, np.ndarray]:
    """Fit per-column params for continuous/count columns.

    The log1p pre-transform is applied when the distribution is both sparse
    (>5% exact zeros) *and* heavy-tailed (max/min of non-zero values > 10).
    The anchor records where the median sits in the normalized [0, 1] range.
    A 101-point quantile grid is computed on *raw* values (before log1p) so
    that typicality at query time works on the natural scale.
    """
    non_null = values[~np.isnan(values)]

    # Distribution-based log1p trigger
    if len(non_null) > 0:
        non_zero = non_null[non_null != 0.0]
        sparsity = float((non_null == 0.0).sum() / len(non_null))
        if len(non_zero) >= 2 and float(non_zero.min()) > 0.0:
            range_ratio = float(non_zero.max()) / float(non_zero.min())
        else:
            range_ratio = 0.0
    else:
        sparsity = 0.0
        range_ratio = 0.0

    use_log1p = (sparsity > 0.05) and (range_ratio > 10.0)

    work_values = np.log1p(values) if use_log1p else values
    work_non_null = np.log1p(non_null) if use_log1p else non_null

    if len(work_non_null) > 0:
        p1 = float(np.percentile(work_non_null, 1))
        p99 = float(np.percentile(work_non_null, 99))
        p50 = float(np.percentile(work_non_null, 50))
    else:
        p1 = p99 = p50 = 0.0

    scale = p99 - p1
    is_constant = scale < _SCALE_EPS
    anchor = 0.5 if is_constant else float(np.clip((p50 - p1) / scale, 0.0, 1.0))

    # Quantile grid on raw (pre-log1p) values. Use nearest-available value at
    # the boundaries when non_null is empty.
    if len(non_null) > 0:
        quantiles = np.quantile(non_null, _QUANTILE_LEVELS).astype(float)
    else:
        quantiles = np.zeros_like(_QUANTILE_LEVELS)

    transformed = _norm(work_values, p1, p99, anchor_fallback=anchor)
    return {
        "scaler_schema_version": _SCALER_SCHEMA_VERSION,
        "kind": kind,
        "use_log1p": use_log1p,
        "p1": p1,
        "p99": p99,
        "anchor": anchor,
        "is_constant": is_constant,
        "quantiles": quantiles,
        "class_freq": None,
    }, transformed


def _apply_type_aware(values: np.ndarray, params: dict) -> np.ndarray:
    work_values = np.log1p(values) if params["use_log1p"] else values
    return _norm(
        work_values,
        params["p1"],
        params["p99"],
        anchor_fallback=params["anchor"],
    )


# ------------------------------------------------------------------
# Binary: pass-through + class_freq
# ------------------------------------------------------------------


def _fit_binary(values: np.ndarray) -> tuple[dict, np.ndarray]:
    """Binary column (values in {0, 1}). Pass-through with class frequencies."""
    non_null = values[~np.isnan(values)]
    n = len(non_null)
    if n > 0:
        p1_class = float((non_null == 1.0).sum() / n)
        p0_class = 1.0 - p1_class
    else:
        p0_class = p1_class = 0.5
    anchor = 1.0 if p1_class >= 0.5 else 0.0
    is_constant = p0_class == 1.0 or p1_class == 1.0
    # NaNs → anchor (majority class) for distance purposes
    out = np.where(np.isnan(values), anchor, values).astype(float)
    return {
        "scaler_schema_version": _SCALER_SCHEMA_VERSION,
        "kind": "binary",
        "use_log1p": False,
        "p1": 0.0,
        "p99": 1.0,
        "anchor": anchor,
        "is_constant": is_constant,
        "quantiles": None,
        "class_freq": {0.0: p0_class, 1.0: p1_class},
    }, out


def _apply_binary(values: np.ndarray, params: dict) -> np.ndarray:
    anchor = params["anchor"]
    return np.where(np.isnan(values), anchor, values).astype(float)


# ------------------------------------------------------------------
# Proportion: pass-through in [0, 1] + quantile grid on raw values
# ------------------------------------------------------------------


def _fit_proportion(values: np.ndarray) -> tuple[dict, np.ndarray]:
    """Proportion column (values already in [0, 1]). Pass-through + quantile grid."""
    non_null = values[~np.isnan(values)]
    if len(non_null) > 0:
        p1 = float(np.percentile(non_null, 1))
        p99 = float(np.percentile(non_null, 99))
        p50 = float(np.percentile(non_null, 50))
        quantiles = np.quantile(non_null, _QUANTILE_LEVELS).astype(float)
    else:
        p1 = p99 = p50 = 0.0
        quantiles = np.zeros_like(_QUANTILE_LEVELS)

    is_constant = (p99 - p1) < _SCALE_EPS
    # For proportions we don't clip: the anchor is the raw median.
    anchor = 0.5 if is_constant else float(np.clip(p50, 0.0, 1.0))

    out = np.where(np.isnan(values), anchor, values)
    out = np.clip(out, 0.0, 1.0).astype(float)
    return {
        "scaler_schema_version": _SCALER_SCHEMA_VERSION,
        "kind": "proportion",
        "use_log1p": False,
        "p1": p1,
        "p99": p99,
        "anchor": anchor,
        "is_constant": is_constant,
        "quantiles": quantiles,
        "class_freq": None,
    }, out


def _apply_proportion(values: np.ndarray, params: dict) -> np.ndarray:
    anchor = params["anchor"]
    out = np.where(np.isnan(values), anchor, values)
    return np.clip(out, 0.0, 1.0).astype(float)


# ------------------------------------------------------------------
# Low-level normalization helper
# ------------------------------------------------------------------


def _norm(values: np.ndarray, p1: float, p99: float, anchor_fallback: float = 0.5) -> np.ndarray:
    """Clip *values* to [p1, p99] and scale to [0, 1].

    NaN inputs are mapped to ``anchor_fallback`` (the normalized median) so
    that downstream L1 distances are not dominated by missing values.
    """
    scale = p99 - p1
    if scale < _SCALE_EPS:
        # Constant column: every observed value maps to the anchor, which
        # we set to 0.5 upstream when scale is degenerate.
        return np.where(np.isnan(values), anchor_fallback, anchor_fallback).astype(float)
    scaled = np.clip((values - p1) / scale, 0.0, 1.0)
    return np.where(np.isnan(values), anchor_fallback, scaled).astype(float)
