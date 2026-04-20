"""Preprocessing pipeline: distribution-aware normalization for mixed-type tabular data.

All columns are mapped to [0, 1] in a way that preserves distributional shape:

- **log1p pre-transform**: applied when the data is sparse (>5% exact zeros)
  *and* the non-zero range spans more than one order of magnitude
  (max/min > 10). This compresses heavy right tails before percentile scaling.
- **Percentile scaling**: clip to the [1st, 99th] percentile range of the
  (optionally log-transformed) reference values and scale to [0, 1].
- **Anchor**: the normalized median ``(p50 - p1) / (p99 - p1)`` is stored per
  column. It naturally falls near 0 for right-skewed / sparse distributions
  and near 0.5 for bell-shaped distributions, enabling downstream richness
  and uniqueness metrics.

The fitted per-column parameters (``use_log1p``, ``p1``, ``p99``, ``anchor``)
are persisted as plain dicts so that queries can be transformed identically
at run time. L1 distance (mean absolute difference across columns) is used
downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eosquality.schema.models import Schema
from eosquality.utils.logging import logger

# Tolerance for treating a scale as zero (constant column).
_SCALE_EPS = 1e-9


class PreprocessPipeline:
    """Fits and applies per-column type-aware normalization.

    Parameters
    ----------
    schema:
        Column definitions that control which columns are processed.
    column_kinds:
        Mapping of column name → detected kind (``"binary"``, ``"count"``,
        ``"continuous"``). Falls back to ``"continuous"`` for any missing key.
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
        # maps col → {"use_log1p": bool, "p1": float, "p99": float, "anchor": float}
        self._scalers: dict[str, dict] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit + transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit normalizers on *df* and return the transformed array."""
        n_cols = len(self._schema.column_names)
        logger.debug(f"Fitting normalization | {n_cols} column(s) | {len(df):,} samples")
        arrays = []
        for col in self._schema.column_names:
            values = df[col].to_numpy(dtype=float)
            params, transformed = _fit_type_aware(values)
            self._scalers[col] = params
            logger.debug(
                f"  {col}: use_log1p={params['use_log1p']}"
                f" | p1={params['p1']:.4g} p99={params['p99']:.4g}"
                f" | anchor={params['anchor']:.3f}"
            )
            arrays.append(transformed)
        self._is_fitted = True
        result = np.column_stack(arrays) if arrays else np.empty((len(df), 0))
        logger.debug(f"Normalization complete | output shape={result.shape}")
        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted normalizers to *df*; returns (n_samples, n_features)."""
        if not self._is_fitted:
            raise RuntimeError(
                "PreprocessPipeline must be fitted before transform()."
            )
        arrays = []
        for col in self._schema.column_names:
            values = df[col].to_numpy(dtype=float)
            arrays.append(_apply_type_aware(values, self._scalers[col]))
        return np.column_stack(arrays) if arrays else np.empty((len(df), 0))

    # ------------------------------------------------------------------
    # Serialization state
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a serialisable state dict for persistence."""
        return {
            "schema": self._schema,
            "scalers": self._scalers,
            "strategy": "type_aware",
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
# Normalization helpers
# ------------------------------------------------------------------


def _fit_type_aware(values: np.ndarray) -> tuple[dict, np.ndarray]:
    """Fit per-column params and return (params, transformed_values).

    The log1p pre-transform is applied when the distribution is both sparse
    (>5% exact zeros) *and* heavy-tailed (max/min of non-zero values > 10).
    The anchor records where the median sits in the normalized [0, 1] range.
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
    anchor = float(np.clip((p50 - p1) / scale, 0.0, 1.0)) if scale >= _SCALE_EPS else 0.5

    transformed = _norm(work_values, p1, p99)
    return {"use_log1p": use_log1p, "p1": p1, "p99": p99, "anchor": anchor}, transformed


def _apply_type_aware(values: np.ndarray, params: dict) -> np.ndarray:
    """Apply pre-fitted distribution-aware normalization to *values*."""
    work_values = np.log1p(values) if params["use_log1p"] else values
    return _norm(work_values, params["p1"], params["p99"])


def _norm(values: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Clip *values* to [p1, p99] and scale to [0, 1]."""
    scale = p99 - p1
    if scale < _SCALE_EPS:
        return np.zeros_like(values)
    return np.clip((values - p1) / scale, 0.0, 1.0)
