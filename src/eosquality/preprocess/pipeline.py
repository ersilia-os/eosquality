"""Preprocessing pipeline: thin wrapper around the eosframes scaler.

eosframes (``eosframes.fit`` / ``eosframes.transform``) owns the per-column
scaling math: it auto-classifies each numeric feature column into one of
seven kinds (``constant``, ``binary``, ``count_zero_mode``, ``count_shifted``,
``continuous_right_skew``, ``continuous_left_skew``, ``continuous_centered``)
and emits a per-kind robust transform into a documented float region inside
``[-1, 1]``.

This module keeps the eosquality-facing surface (``fit_transform``,
``transform``, ``raw_numeric_values``, ``get_state``, ``from_state``) so the
quality API and scoring code don't need to change. State persists as the
plain JSON-serialisable dict returned by ``eosframes.fit`` — see
:mod:`eosquality.io.save` / :mod:`eosquality.io.load`.

Typicality consumes the int8-quantized output of the same scaler (see
:mod:`eosquality.scoring.typicality`) so no separate CDF artifact is fit
or persisted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import eosframes

from eosquality.schema.models import Schema
from eosquality.utils.logging import logger


class PreprocessPipeline:
    """Fits and applies the eosframes scaler over the schema's feature columns.

    Parameters
    ----------
    schema:
        Column definitions that control which columns are processed.
    """

    def __init__(self, schema: Schema) -> None:
        self._schema = schema
        self._params: dict | None = None

    # ------------------------------------------------------------------
    # Fit + transform
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the eosframes scaler on *df* and return the scaled feature array."""
        feature_cols = list(self._schema.column_names)
        logger.debug(
            f"Fitting normalization (eosframes) | {len(feature_cols)} column(s) | "
            f"{len(df):,} samples"
        )
        feature_df = df[feature_cols]
        self._params = eosframes.fit(feature_df)
        if logger.verbose:
            kind_counts: dict[str, int] = {}
            for entry in self._params["columns"].values():
                kind = entry["transform"]["kind"]
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            logger.debug(
                "  kind breakdown: "
                + ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items()))
            )
        result = self._transform_to_array(feature_df)
        logger.debug(f"Normalization complete | output shape={result.shape}")
        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted eosframes scaler to *df*; returns the feature array."""
        if self._params is None:
            raise RuntimeError(
                "PreprocessPipeline must be fitted before transform()."
            )
        feature_cols = list(self._schema.column_names)
        return self._transform_to_array(df[feature_cols])

    def raw_numeric_values(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw float values for the fitted columns, shape (n, n_features).

        Typicality re-quantizes the eosframes scaled output rather than reading
        the raw values, but the kNN distance fallback and external callers may
        still want the unscaled array.
        """
        if self._params is None:
            raise RuntimeError(
                "PreprocessPipeline must be fitted before raw_numeric_values()."
            )
        feature_cols = list(self._schema.column_names)
        if not feature_cols:
            return np.empty((len(df), 0))
        return df[feature_cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Serialization state
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a serialisable state dict for persistence."""
        if self._params is None:
            raise RuntimeError("PreprocessPipeline must be fitted before get_state().")
        return {
            "schema": self._schema,
            "scaler_params": self._params,
        }

    @classmethod
    def from_state(cls, state: dict) -> "PreprocessPipeline":
        """Reconstruct a fitted pipeline from a persisted state dict."""
        pipeline = cls(schema=state["schema"])
        pipeline._params = state["scaler_params"]
        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform_to_array(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Apply ``eosframes.transform`` and return the values as float64.

        eosframes returns float32 by default; the kNN distance code downstream
        works in float64, so we widen here once instead of every call site.
        """
        assert self._params is not None
        scaled = eosframes.transform(
            feature_df, self._params, output_dtype="float32"
        )
        feature_cols = list(self._schema.column_names)
        if not feature_cols:
            return np.empty((len(feature_df), 0))
        return scaled[feature_cols].to_numpy(dtype=np.float64)
