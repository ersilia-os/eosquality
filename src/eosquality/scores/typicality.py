"""Typicality score: per-feature + CDF-calibrated aggregate, no library needed.

Typicality is **density-based**: for each query value, look up its int8
quantization in the per-column count LUT built on the reference and
return ``count(int8) / max_count``.

This handles every distribution shape uniformly — unimodal, multimodal,
constant, binary — with no kind dispatch: the most common int8 always
scores typicality 1.0, every other int8 scores in proportion to how
often it appears in the reference, and unseen int8 values score 0.

Missing values carry no information: a NaN feature stays NaN per feature
and is ignored by the aggregate; a row whose every feature is NaN scores
NaN.

Two columns are published for the whole model:

- ``ref_typicality_raw``: the **66th percentile** across features of the
  per-feature values (``AGGREGATE_QUANTILE``), in [0, 1].
- ``ref_typicality_pct``: each feature's value is first placed on that
  column's own reference distribution of the same density (mid-rank
  percentile, so columns with different shapes weigh equally; higher is
  more typical), the per-feature percentiles are aggregated at the same
  66th percentile, and that aggregate is mapped through the reference's own
  sorted distribution of it (:func:`_score_from_aggregates`). The result is
  uniform under the reference and comparable across models with different
  feature counts. The per-feature raw values and percentiles are returned as
  ``per_feature`` and ``per_feature_pct``.

A column's density takes at most 256 values, so its percentile table is
derived exactly from the count LUT (:func:`percentile_luts`) and is not saved.
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
    _aggregate_percentiles,
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

    score: pd.Series  # (n_query,) whole-model typicality percentile, in (0, 1]
    score_raw: pd.Series  # (n_query,) Q66 of the per-feature values, in [0, 1]
    per_feature: pd.DataFrame  # (n_query, n_features) count / max count
    per_feature_pct: pd.DataFrame  # (n_query, n_features) per-column percentiles
    metadata: dict[str, Any] = field(default_factory=dict)


class Typicality(ScoreComponent):
    """Density-based per-feature typicality scorer.

    Fitted state:

    - per-column int8 count LUTs, ``(256, n_features)``: reference counts per
      level per column, built at fit time (saved);
    - ``pct_luts_`` — ``(256, n_features)`` per-column percentile of each
      level, derived from the counts (not saved);
    - the ascending reference per-row Q66 aggregates of the per-column
      percentiles: the CDF table that maps the aggregate to the score;
    - ``reference_typicality_`` — mean reference-as-query calibrated
      typicality. ≈ 0.5 by construction; a sanity-check anchor.

    Depends only on :class:`SharedFitState` — no reference library required.
    """

    NAME = SUBFOLDER

    def __init__(self) -> None:
        super().__init__()
        self._count_luts: np.ndarray | None = None  # (256, n_features)
        self._pct_luts: np.ndarray | None = None  # (256, n_features), derived
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
        pct_luts = percentile_luts(count_luts)
        ref_agg = _aggregate_percentiles(lookup_percentiles(ref_scaled, pct_luts))
        sorted_self_aggregates = _sorted_finite(ref_agg, "Typicality")

        self._shared = shared
        self._count_luts = count_luts
        self._pct_luts = pct_luts
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
        assert self._pct_luts is not None
        assert self._sorted_self_aggregates is not None

        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)

        per_feature, raw_aggregate = compute_typicality(
            scaled_values=query_repr,
            count_luts=self._count_luts,
        )
        per_feature_pct = lookup_percentiles(query_repr, self._pct_luts)
        score = _score_from_aggregates(
            _aggregate_percentiles(per_feature_pct), self._sorted_self_aggregates
        )
        idx = list(query.index)
        columns = list(self._shared.selected_columns)
        return TypicalityRunResult(
            score=pd.Series(score, index=idx, name="ref_typicality_pct"),
            score_raw=pd.Series(raw_aggregate, index=idx, name="ref_typicality_raw"),
            per_feature=pd.DataFrame(per_feature, index=idx, columns=columns),
            per_feature_pct=pd.DataFrame(per_feature_pct, index=idx, columns=columns),
            metadata={
                "anchor": self._reference_typicality,
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
        self._pct_luts = percentile_luts(self._count_luts)
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
            and self._pct_luts is not None
            and self._sorted_self_aggregates is not None
            and self._reference_typicality is not None
        )

    @property
    def pct_luts_(self) -> np.ndarray:
        """Per-column percentile of each int8 level (derived from the counts).

        Returns
        -------
        numpy.ndarray
            ``(256, n_features)``.
        """
        self._check_fitted()
        assert self._pct_luts is not None
        return self._pct_luts

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


def percentile_luts(count_luts: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each level's density among the reference values.

    Within a column the density of a level is its count (over a constant),
    so a level's percentile is the share of reference values whose level is
    less common, with ties at half weight, as in
    :func:`~eosquality.scores._helpers._cdf_score`: the most common level
    scores near 1, rare levels near 0, and unseen levels the floor
    ``1 / (2 n)``. A column with no counted value gives NaN.

    Parameters
    ----------
    count_luts : numpy.ndarray
        ``(256, n_features)`` counts, as returned by :func:`fit_typicality_luts`.

    Returns
    -------
    numpy.ndarray
        ``(256, n_features)`` percentiles in ``(0, 1]``.
    """
    counts = np.asarray(count_luts, dtype=np.float64)
    out = np.full(counts.shape, np.nan)
    for j in range(counts.shape[1]):
        c = counts[:, j]
        n = c.sum()
        if n <= 0:
            continue
        below = (c[None, :] * (c[None, :] < c[:, None])).sum(axis=1)
        at_or_below = (c[None, :] * (c[None, :] <= c[:, None])).sum(axis=1)
        out[:, j] = np.clip((below + at_or_below) / (2.0 * n), 0.5 / n, 1.0)
    return out


def lookup_percentiles(scaled_values: np.ndarray, pct_luts: np.ndarray) -> np.ndarray:
    """Per-feature typicality percentiles of scaled values.

    Parameters
    ----------
    scaled_values : numpy.ndarray
        ``(n, n_features)`` eosframes-scaled values.
    pct_luts : numpy.ndarray
        ``(256, n_features)`` from :func:`percentile_luts`.

    Returns
    -------
    numpy.ndarray
        ``(n, n_features)`` percentiles; NaN where the input is NaN.
    """
    n_features = scaled_values.shape[1] if scaled_values.ndim > 1 else 0
    if n_features == 0:
        return np.ones((scaled_values.shape[0], 0))
    q_int8 = _quantize_to_int8(scaled_values)
    out = pct_luts[q_int8 + _LUT_OFFSET, np.arange(n_features)[None, :]]
    out[q_int8 == -_LUT_OFFSET] = np.nan
    return out
