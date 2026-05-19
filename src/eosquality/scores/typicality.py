"""Typicality score: per-feature + aggregate, no vector index needed.

Typicality is **density-based**: for each query value, look up its int8
quantization in the per-column count LUT built on the reference and
return ``count(int8) / max_count``.

This handles every distribution shape uniformly — unimodal, multimodal,
constant, binary — with no kind dispatch: the most common int8 always
scores typicality 1.0, every other int8 scores in proportion to how
often it appears in the reference, and unseen int8 values score 0.
NaN queries return typicality 1.0 (no information).

The aggregate per query is the **arithmetic mean** of the per-feature
typicality values. The single ``typicality`` column is the user-facing
output; per-feature values are retained on :class:`TypicalityRunResult`
for diagnostic tools but not surfaced by :class:`ErsiliaQuality`.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import _component_metadata, _make_pipeline
from eosquality.shared.fit import fit_shared
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


_INT8_MAX_VAL = 127
_LUT_SIZE = 256
_LUT_OFFSET = 128  # lut index = int8 + offset; the slot at index 0 is the NaN sentinel
SUBFOLDER = "typicality"
STATE_FILE = "state.json"
METADATA_FILE = "metadata.json"


@dataclass
class TypicalityRunResult:
    """Result returned by :meth:`Typicality.run`."""

    score: pd.Series  # (n_query,) aggregate typicality
    per_feature: pd.DataFrame  # (n_query, n_features)
    metadata: dict[str, Any] = field(default_factory=dict)


class Typicality:
    """Density-based per-feature typicality scorer.

    Holds two pieces of fitted state:

    - ``count_luts_`` — ``(256, n_features)`` int array of reference counts
      per int8 level per column. Built once at fit time and consulted at
      query time.
    - ``reference_typicality_`` — mean aggregate typicality of the reference
      under its own LUTs. A calibration anchor for downstream readers.

    Depends only on :class:`SharedFitState` — no vector index required.
    """

    def __init__(self) -> None:
        self._shared: SharedFitState | None = None
        self._count_luts: np.ndarray | None = None  # (256, n_features)
        self._reference_typicality: float | None = None
        self._fit_duration_seconds: float | None = None
        self._fit_timestamp: str | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        *,
        eos_id: str | None = None,
        version: str | None = None,
        shared: SharedFitState | None = None,
    ) -> "Typicality":
        """Fit on a reference DataFrame.

        Scales the reference with the shared eosframes pipeline, builds the
        per-column int8 count LUTs (:func:`fit_typicality_luts`), and
        records ``reference_typicality_`` — the mean aggregate typicality
        of the reference under its own LUTs — as a calibration baseline.

        Either pass a pre-fit ``shared=`` (when composed by ErsiliaQuality),
        or pass ``eos_id`` + ``version`` so Typicality can fit the shared
        state itself.

        Records the wall-clock duration and a UTC timestamp; both are
        persisted by :meth:`save` to ``typicality/metadata.json``.
        """
        t0 = time.perf_counter()
        if shared is None:
            if eos_id is None or version is None:
                raise ValueError(
                    "Typicality.fit needs either a pre-fit shared= argument, "
                    "or eos_id= and version= so it can fit the shared state itself."
                )
            shared, _ref_repr = fit_shared(reference, eos_id=eos_id, version=version)
        else:
            validate_against_schema(reference, shared.schema)

        ref_scaled = _make_pipeline(shared).transform(reference)
        count_luts = fit_typicality_luts(ref_scaled)
        _, ref_agg = compute_typicality(
            scaled_values=ref_scaled,
            count_luts=count_luts,
        )

        self._shared = shared
        self._count_luts = count_luts
        self._reference_typicality = float(np.mean(ref_agg))
        self._fit_duration_seconds = float(time.perf_counter() - t0)
        self._fit_timestamp = datetime.now(tz=timezone.utc).isoformat()
        logger.debug(
            f"Typicality fit | reference_typicality={self._reference_typicality:.4f}"
            f" | duration={self._fit_duration_seconds:.3f}s"
        )
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        query: pd.DataFrame,
        *,
        query_repr: np.ndarray | None = None,
    ) -> TypicalityRunResult:
        """Score query samples.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference.
        query_repr:
            Optional pre-scaled query array (output of the eosframes scaler)
            ``(n_query, n_features)``. If provided, the eosframes transform
            step is skipped — used by ErsiliaQuality to share scaling work
            across multiple scores.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._count_luts is not None

        validate_against_schema(query, self._shared.schema)

        if query_repr is None:
            query_repr = _make_pipeline(self._shared).transform(query)

        per_feature, score = compute_typicality(
            scaled_values=query_repr,
            count_luts=self._count_luts,
        )
        per_feature_df = pd.DataFrame(
            per_feature,
            index=list(query.index),
            columns=list(self._shared.schema.column_names),
        )
        score_series = pd.Series(score, index=list(query.index), name="typicality")
        return TypicalityRunResult(
            score=score_series,
            per_feature=per_feature_df,
            metadata={
                "reference_typicality": self._reference_typicality,
                "n_reference": len(self._shared.reference_ids),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, root: str | pathlib.Path) -> pathlib.Path:
        """Persist into ``<root>/shared/`` and ``<root>/typicality/``.

        Writes two files under ``typicality/``:

        - ``state.json`` — the ``reference_typicality`` baseline and the
          per-column ``count_luts`` keyed by column name.
        - ``metadata.json`` — fit timestamp, fit duration, n_samples,
          n_features, eosquality_version.

        Also writes the shared subfolder via :func:`save_shared` so the
        artifact is self-contained.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._count_luts is not None
        save_shared(self._shared, root)
        folder = pathlib.Path(root) / SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)
        column_names = list(self._shared.schema.column_names)
        payload = {
            "reference_typicality": self._reference_typicality,
            "column_names": column_names,
            "count_luts": {
                col: self._count_luts[:, j].astype(int).tolist()
                for j, col in enumerate(column_names)
            },
        }
        with open(folder / STATE_FILE, "w") as f:
            json.dump(payload, f)
        meta = _component_metadata(
            component="typicality",
            k=None,
            fit_timestamp=self._fit_timestamp,
            fit_duration_seconds=self._fit_duration_seconds,
        )
        with open(folder / METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug(
            f"  typicality/ | reference_typicality={self._reference_typicality:.4f}"
            f" | count_luts: {len(column_names)} columns × {_LUT_SIZE} levels"
            f" | fit_duration={meta['fit_duration_seconds']:.3f}s"
        )
        return pathlib.Path(root)

    @classmethod
    def load(cls, root: str | pathlib.Path) -> "Typicality":
        """Reconstruct from ``<root>/shared/`` + ``<root>/typicality/state.json``."""
        shared = load_shared(root)
        folder = pathlib.Path(root) / SUBFOLDER
        with open(folder / STATE_FILE) as f:
            payload = json.load(f)

        column_names = list(shared.schema.column_names)
        if payload["column_names"] != column_names:
            raise ValueError(
                f"typicality/{STATE_FILE} column_names do not match shared schema."
            )
        count_luts = np.zeros((_LUT_SIZE, len(column_names)), dtype=np.int64)
        for j, col in enumerate(column_names):
            count_luts[:, j] = np.asarray(payload["count_luts"][col], dtype=np.int64)

        meta_path = folder / METADATA_FILE
        fit_duration = None
        fit_timestamp = None
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            fit_duration = float(meta.get("fit_duration_seconds", 0.0))
            fit_timestamp = meta.get("fit_timestamp")

        instance = cls()
        instance._shared = shared
        instance._count_luts = count_luts
        instance._reference_typicality = float(payload["reference_typicality"])
        instance._fit_duration_seconds = fit_duration
        instance._fit_timestamp = fit_timestamp
        return instance

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._count_luts is not None
            and self._reference_typicality is not None
        )

    @property
    def shared_(self) -> SharedFitState:
        self._check_fitted()
        assert self._shared is not None
        return self._shared

    @property
    def count_luts_(self) -> np.ndarray:
        self._check_fitted()
        assert self._count_luts is not None
        return self._count_luts

    @property
    def reference_typicality_(self) -> float:
        self._check_fitted()
        assert self._reference_typicality is not None
        return self._reference_typicality

    @property
    def fit_duration_seconds_(self) -> float | None:
        return self._fit_duration_seconds

    @property
    def fit_timestamp_(self) -> str | None:
        return self._fit_timestamp

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Typicality must be fitted (or loaded) before use.")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _quantize_to_int8(scaled: np.ndarray) -> np.ndarray:
    """Mirror the eosframes int8 quantization. NaN → sentinel -128.

    Returns int64 values in ``[-128, 127]`` (one wider than the int8 range
    because we use ``-128`` as the NaN sentinel). Use ``+ 128`` to index the
    256-slot LUT.
    """
    nan_mask = np.isnan(scaled)
    q = np.where(nan_mask, -_LUT_OFFSET, np.round(scaled * _INT8_MAX_VAL))
    q = np.clip(q, -_LUT_OFFSET, _INT8_MAX_VAL)
    return q.astype(np.int64)


def fit_typicality_luts(scaled_reference: np.ndarray) -> np.ndarray:
    """Build per-column int8 count LUTs from reference scaled values.

    Returns
    -------
    count_luts: np.ndarray
        Shape ``(256, n_features)``; ``count_luts[int8 + 128, j]`` is the
        number of reference rows whose feature ``j`` quantized to ``int8``.
        The NaN-sentinel slot (index 0) is always 0 — NaN reference rows are
        excluded from the count.
    """
    n_features = scaled_reference.shape[1] if scaled_reference.ndim > 1 else 0
    luts = np.zeros((_LUT_SIZE, n_features), dtype=np.int64)
    if n_features == 0 or scaled_reference.shape[0] == 0:
        return luts
    ref_int8 = _quantize_to_int8(scaled_reference)
    for j in range(n_features):
        col = ref_int8[:, j]
        valid = col != -_LUT_OFFSET
        if valid.any():
            np.add.at(luts[:, j], col[valid] + _LUT_OFFSET, 1)
    return luts


def compute_typicality(
    scaled_values: np.ndarray,
    count_luts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature and aggregate typicality from per-column count LUTs.

    Parameters
    ----------
    scaled_values:
        ``(n_query, n_features)`` float array produced by the eosframes
        scaler (the output of :meth:`PreprocessPipeline.transform`).
    count_luts:
        ``(256, n_features)`` int array of reference counts per int8 level,
        as returned by :func:`fit_typicality_luts`. Column order must match
        ``scaled_values``.

    Returns
    -------
    per_feature:
        ``(n_query, n_features)`` typicality in ``[0, 1]``.
    aggregate:
        ``(n_query,)`` arithmetic mean of ``per_feature`` across features.
    """
    n_query = scaled_values.shape[0]
    n_features = scaled_values.shape[1] if scaled_values.ndim > 1 else 0
    if n_features == 0:
        return np.ones((n_query, 0)), np.ones(n_query)
    if count_luts.shape != (_LUT_SIZE, n_features):
        raise ValueError(
            f"count_luts must have shape ({_LUT_SIZE}, {n_features}); "
            f"got {count_luts.shape}."
        )

    q_int8 = _quantize_to_int8(scaled_values)
    per_feature = np.empty((n_query, n_features), dtype=np.float64)
    for j in range(n_features):
        col_lut = count_luts[:, j]
        col_max = col_lut.max()
        max_count = float(col_max) if col_max > 0 else 1.0

        q_col = q_int8[:, j]
        nan_q = q_col == -_LUT_OFFSET
        typ = col_lut[q_col + _LUT_OFFSET] / max_count
        per_feature[:, j] = np.where(nan_q, 1.0, typ)

    aggregate = per_feature.mean(axis=1)
    return per_feature, aggregate
