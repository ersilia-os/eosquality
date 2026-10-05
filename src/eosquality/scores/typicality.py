"""Typicality score: per-feature + CDF-calibrated aggregate, no vector index needed.

Typicality is **density-based**: for each query value, look up its int8
quantization in the per-column count LUT built on the reference and
return ``count(int8) / max_count``.

This handles every distribution shape uniformly — unimodal, multimodal,
constant, binary — with no kind dispatch: the most common int8 always
scores typicality 1.0, every other int8 scores in proportion to how
often it appears in the reference, and unseen int8 values score 0.

Missing values carry no information: a NaN feature stays NaN per feature
and is ignored by the aggregate; a row whose every feature is NaN scores
NaN. The per-row aggregate is the **66th percentile** of the finite
per-feature values (``AGGREGATE_QUANTILE``), mapped through the
reference's own sorted Q66 distribution via :func:`_score_from_aggregates`
(mid-rank CDF), so the reference scores ~Uniform(0, 1) and the score is
comparable across models with different feature counts. Per-feature
values are retained on :class:`TypicalityRunResult` for diagnostics.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.schema.infer import validate_against_schema
from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import (
    _make_query_repr,
    _nan_aggregate,
    _reference_repr,
    _resolve_shared,
    _score_from_aggregates,
    _sorted_finite,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger

_INT8_MAX_VAL = 127
_LUT_SIZE = 256
_LUT_OFFSET = 128  # lut index = int8 + offset; the slot at index 0 is the NaN sentinel
SUBFOLDER = "typicality"
STATE_FILE = "state.json"
SELF_AGGREGATES_FILE = "reference_self_aggregates.npy"


@dataclass
class TypicalityRunResult:
    """Result returned by :meth:`Typicality.run`."""

    score: pd.Series  # (n_query,) calibrated aggregate typicality in [0, 1]
    score_raw: pd.Series  # (n_query,) Q66 aggregate before CDF lookup, in [0, 1]
    per_feature: pd.DataFrame  # (n_query, n_features)
    metadata: dict[str, Any] = field(default_factory=dict)


class Typicality(ScoreComponent):
    """Density-based per-feature typicality scorer.

    Holds three pieces of fitted state:

    - ``count_luts_`` — ``(256, n_features)`` int array of reference counts
      per int8 level per column. Built once at fit time and consulted at
      query time.
    - ``sorted_self_aggregates_`` — ``(n_ref,)`` ascending array of
      reference per-row Q66 aggregates. The CDF lookup table that maps
      raw aggregates to calibrated scores.
    - ``reference_typicality_`` — mean reference-as-query calibrated
      typicality. ≈ 0.5 by construction; a sanity-check anchor.

    Depends only on :class:`SharedFitState` — no vector index required.
    """

    NAME = SUBFOLDER

    def __init__(self) -> None:
        super().__init__()
        self._count_luts: np.ndarray | None = None  # (256, n_features)
        self._sorted_self_aggregates: np.ndarray | None = None  # (n_ref,)
        self._reference_typicality: float | None = None

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
    ) -> Typicality:
        """Build the int8 density LUTs and the Q66 calibration table.

        Parameters
        ----------
        reference : pandas.DataFrame
            Predictions on the reference library.
        eos_id, version : str, optional
            Model id and version, needed only to fit ``shared`` here.
        shared : SharedFitState, optional
            Pre-fit shared state (as passed by :class:`ErsiliaQuality`).

        Returns
        -------
        Typicality
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        shared = _resolve_shared(
            reference,
            shared=shared,
            eos_id=eos_id,
            version=version,
            component="Typicality",
        )
        ref_scaled = _reference_repr(shared, reference)
        count_luts = fit_typicality_luts(ref_scaled)
        _, ref_agg = compute_typicality(scaled_values=ref_scaled, count_luts=count_luts)
        sorted_self_aggregates = _sorted_finite(ref_agg, "Typicality")

        self._shared = shared
        self._count_luts = count_luts
        self._sorted_self_aggregates = sorted_self_aggregates
        self._reference_typicality = float(
            np.nanmean(_score_from_aggregates(ref_agg, sorted_self_aggregates))
        )
        self._finish_fit(t0)
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
        query : pandas.DataFrame
            The reference's numeric output columns.
        query_repr : numpy.ndarray, optional
            Pre-scaled, feature-selected query array; skips validation and scaling
            (the orchestrator shares this work across scores).

        Returns
        -------
        TypicalityRunResult
            Calibrated score, Q66 aggregate, per-feature values and metadata.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._count_luts is not None
        assert self._sorted_self_aggregates is not None

        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)

        per_feature, raw_aggregate = compute_typicality(
            scaled_values=query_repr,
            count_luts=self._count_luts,
        )
        n_ref = len(self._shared.reference_ids)
        score = _score_from_aggregates(raw_aggregate, self._sorted_self_aggregates)
        idx = list(query.index)
        return TypicalityRunResult(
            score=pd.Series(score, index=idx, name="typicality"),
            score_raw=pd.Series(raw_aggregate, index=idx, name="typicality_raw"),
            per_feature=pd.DataFrame(
                per_feature, index=idx, columns=list(self._shared.selected_columns)
            ),
            metadata={
                "reference_typicality": self._reference_typicality,
                "n_reference": n_ref,
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline + per-column LUTs) and the CDF array."""
        assert self._shared is not None
        assert self._count_luts is not None
        assert self._sorted_self_aggregates is not None
        np.save(folder / SELF_AGGREGATES_FILE, self._sorted_self_aggregates)
        column_names = list(self._shared.selected_columns)
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

    def _load_own(self, folder: pathlib.Path) -> None:
        assert self._shared is not None
        payload = read_json(folder / STATE_FILE, self.NAME)
        column_names = list(self._shared.selected_columns)
        if payload["column_names"] != column_names:
            raise ValueError(
                f"typicality/{STATE_FILE} column_names do not match "
                "shared selected columns."
            )
        self._count_luts = np.stack(
            [
                np.asarray(payload["count_luts"][col], dtype=np.int64)
                for col in column_names
            ],
            axis=1,
        ).reshape(_LUT_SIZE, len(column_names))
        self._sorted_self_aggregates = np.load(
            require_file(folder / SELF_AGGREGATES_FILE, self.NAME)
        )
        self._reference_typicality = float(payload["reference_typicality"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return (
            self._shared is not None
            and self._count_luts is not None
            and self._sorted_self_aggregates is not None
            and self._reference_typicality is not None
        )

    @property
    def count_luts_(self) -> np.ndarray:
        """Per-column int8 count LUTs.

        Returns
        -------
        numpy.ndarray
            ``(256, n_features)``.
        """
        self._check_fitted()
        assert self._count_luts is not None
        return self._count_luts

    @property
    def sorted_self_aggregates_(self) -> np.ndarray:
        """Sorted reference Q66 aggregates (the calibration CDF).

        Returns
        -------
        numpy.ndarray
        """
        self._check_fitted()
        assert self._sorted_self_aggregates is not None
        return self._sorted_self_aggregates

    @property
    def reference_typicality_(self) -> float:
        """Mean calibrated typicality of the reference (about 0.5).

        Returns
        -------
        float
        """
        self._check_fitted()
        assert self._reference_typicality is not None
        return self._reference_typicality


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _quantize_to_int8(scaled: np.ndarray) -> np.ndarray:
    """Mirror the eosframes int8 quantization. NaN → sentinel -128.

    Finite values are rounded to ``[-127, 127]``; ``-128`` is reserved for
    NaN, so an out-of-range finite value can never be mistaken for a
    missing one. Use ``+ 128`` to index the 256-slot LUT.
    """
    q = np.clip(
        np.round(np.nan_to_num(scaled) * _INT8_MAX_VAL), -_INT8_MAX_VAL, _INT8_MAX_VAL
    )
    q = np.where(np.isnan(scaled), -_LUT_OFFSET, q)
    return q.astype(np.int64)


def fit_typicality_luts(scaled_reference: np.ndarray) -> np.ndarray:
    """Build per-column int8 count LUTs from reference scaled values.

    Parameters
    ----------
    scaled_reference : numpy.ndarray
        ``(n_ref, n_features)`` eosframes-scaled reference values.

    Returns
    -------
    numpy.ndarray
        ``(256, n_features)`` counts; ``luts[int8 + 128, j]`` is the number of
        reference rows whose feature ``j`` quantises to ``int8``. Slot 0 (the
        NaN sentinel) stays 0: NaN reference values are not counted.
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
        ``(n_query, n_features)`` typicality in ``[0, 1]``; NaN where the
        input is NaN.
    aggregate:
        ``(n_query,)`` 66th-percentile of the finite ``per_feature`` values
        (see ``AGGREGATE_QUANTILE``); NaN for all-NaN rows. The shift away
        from the mean prevents the aggregate from collapsing to the
        per-feature expectation as ``n_features`` grows; downstream CDF
        calibration re-spreads it to uniform under the reference.
    """
    n_query = scaled_values.shape[0]
    n_features = scaled_values.shape[1] if scaled_values.ndim > 1 else 0
    if n_features == 0:
        return np.ones((n_query, 0)), np.full(n_query, np.nan)
    if count_luts.shape != (_LUT_SIZE, n_features):
        raise ValueError(
            f"count_luts must have shape ({_LUT_SIZE}, {n_features}); "
            f"got {count_luts.shape}."
        )

    q_int8 = _quantize_to_int8(scaled_values)
    col_max = count_luts.max(axis=0).astype(np.float64)
    col_max[col_max <= 0] = 1.0
    counts = count_luts[q_int8 + _LUT_OFFSET, np.arange(n_features)[None, :]]
    per_feature = counts / col_max[None, :]
    per_feature[q_int8 == -_LUT_OFFSET] = np.nan
    return per_feature, _nan_aggregate(per_feature)
