"""Extremity score: how far each scaled value sits from the per-column center.

Extremity is position-based: per-feature extremity is the absolute value
of the eosframes-scaled value, clipped at 1.0. The most central value
(scaled to 0) scores 0; values at the rails (scaled to ±1 or beyond)
score 1. The sign is deliberately dropped: the score measures distance
from the centre, not direction. Complementary to typicality (which is
density-based, not position-based) — the pair (extremity, typicality)
describes a query better than either alone.

Missing values carry no information: a NaN feature stays NaN per feature
and is ignored by every aggregate; a row whose every feature is NaN scores
NaN (same policy as typicality).

Two columns are published for the whole model:

- ``ref_extremity_raw``: the **66th percentile** across features of the
  per-feature values (``AGGREGATE_QUANTILE``), in [0, 1].
- ``ref_extremity_pct``: each feature's value is first placed on that
  column's own reference distribution (mid-rank percentile, so columns with
  different shapes weigh equally), the per-feature percentiles are
  aggregated at the same 66th percentile, and that aggregate is mapped
  through the reference's own sorted distribution of it. The result is
  uniform under the reference and comparable across models with different
  feature counts. The per-feature raw values and percentiles are returned
  as ``per_feature`` and ``per_feature_pct``.
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
    _cdf_score,
    _make_query_repr,
    _nan_aggregate,
    _reference_repr,
    _resolve_shared,
    _score_from_aggregates,
    _sorted_finite,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger

SUBFOLDER = "extremity"
STATE_FILE = "state.json"
SELF_AGGREGATES_FILE = "reference_self_aggregates.npy"
COLUMN_TABLES_FILE = "column_tables.npz"


@dataclass
class ExtremityRunResult:
    """Result returned by :meth:`Extremity.run`."""

    score: pd.Series  # (n_query,) percentile of the raw value in the reference
    score_raw: pd.Series  # (n_query,) Q66 aggregate before CDF lookup, in [0, 1]
    per_feature: pd.DataFrame  # (n_query, n_features) min(|scaled|, 1)
    per_feature_pct: pd.DataFrame  # (n_query, n_features) percentile per column
    metadata: dict[str, Any] = field(default_factory=dict)


class Extremity(ScoreComponent):
    """Position-based per-feature extremity scorer.

    Holds three pieces of fitted state:

    - ``column_tables_`` — per selected column, the ascending reference
      per-feature values (float32): the table of each column's percentile.
    - ``sorted_self_aggregates_`` — ``(n_ref,)`` ascending array of the
      reference per-row Q66 of the per-feature percentiles. The CDF lookup
      table that maps that aggregate to the calibrated score.
    - ``reference_extremity_`` — mean reference-as-query calibrated
      extremity. ≈ 0.5 by construction; a sanity-check anchor.

    Depends only on :class:`SharedFitState` — no reference library required, no
    per-column LUTs.
    """

    NAME = SUBFOLDER

    def __init__(self) -> None:
        super().__init__()
        self._sorted_self_aggregates: np.ndarray | None = None  # (n_ref,)
        self._column_tables: dict[str, np.ndarray] | None = None
        self._reference_extremity: float | None = None

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
    ) -> Extremity:
        """Build the Q66 calibration table of reference extremity.

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
        Extremity
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        shared = _resolve_shared(
            reference,
            shared=shared,
            eos_id=eos_id,
            version=version,
            component="Extremity",
        )
        ref_per_feature, _ = compute_extremity(
            scaled_values=_reference_repr(shared, reference)
        )
        tables = {
            name: _column_table(ref_per_feature[:, j])
            for j, name in enumerate(shared.selected_columns)
        }
        ref_agg = _aggregate_percentiles(
            _per_column_percentiles(ref_per_feature, tables)
        )
        sorted_self_aggregates = _sorted_finite(ref_agg, "Extremity")

        self._shared = shared
        self._column_tables = tables
        self._sorted_self_aggregates = sorted_self_aggregates
        self._reference_extremity = float(
            np.nanmean(_score_from_aggregates(ref_agg, sorted_self_aggregates))
        )
        self._finish_fit(t0)
        logger.debug(
            f"Extremity fit | reference_extremity={self._reference_extremity:.4f}"
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
    ) -> ExtremityRunResult:
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
        ExtremityRunResult
            Calibrated score, Q66 aggregate, per-feature values and metadata.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._sorted_self_aggregates is not None
        assert self._column_tables is not None

        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)

        per_feature, raw_aggregate = compute_extremity(scaled_values=query_repr)
        per_feature_pct = _per_column_percentiles(per_feature, self._column_tables)
        score = _score_from_aggregates(
            _aggregate_percentiles(per_feature_pct), self._sorted_self_aggregates
        )
        idx = list(query.index)
        columns = list(self._shared.selected_columns)
        return ExtremityRunResult(
            score=pd.Series(score, index=idx, name="ref_extremity_pct"),
            score_raw=pd.Series(raw_aggregate, index=idx, name="ref_extremity_raw"),
            per_feature=pd.DataFrame(per_feature, index=idx, columns=columns),
            per_feature_pct=pd.DataFrame(per_feature_pct, index=idx, columns=columns),
            metadata={
                "anchor": self._reference_extremity,
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline), the column tables and the CDF array."""
        assert self._sorted_self_aggregates is not None
        assert self._column_tables is not None
        np.save(folder / SELF_AGGREGATES_FILE, self._sorted_self_aggregates)
        names = list(self._column_tables)
        np.savez(
            folder / COLUMN_TABLES_FILE,
            **{f"c{j:03d}": self._column_tables[n] for j, n in enumerate(names)},
        )
        with open(folder / STATE_FILE, "w") as f:
            json.dump(
                {
                    "reference_extremity": self._reference_extremity,
                    "columns": names,
                },
                f,
            )

    def _load_own(self, folder: pathlib.Path) -> None:
        payload = read_json(folder / STATE_FILE, self.NAME)
        self._sorted_self_aggregates = np.load(
            require_file(folder / SELF_AGGREGATES_FILE, self.NAME)
        )
        self._reference_extremity = float(payload["reference_extremity"])
        with np.load(require_file(folder / COLUMN_TABLES_FILE, self.NAME)) as npz:
            self._column_tables = {
                name: npz[f"c{j:03d}"] for j, name in enumerate(payload["columns"])
            }

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
            and self._sorted_self_aggregates is not None
            and self._column_tables is not None
            and self._reference_extremity is not None
        )

    @property
    def column_tables_(self) -> dict[str, np.ndarray]:
        """Ascending reference per-feature values of each selected column.

        Returns
        -------
        dict of str to numpy.ndarray
        """
        self._check_fitted()
        assert self._column_tables is not None
        return self._column_tables

    @property
    def sorted_self_aggregates_(self) -> np.ndarray:
        """Sorted reference Q66 aggregates of the percentiles (the final CDF).

        Returns
        -------
        numpy.ndarray
        """
        self._check_fitted()
        assert self._sorted_self_aggregates is not None
        return self._sorted_self_aggregates

    @property
    def reference_extremity_(self) -> float:
        """Mean calibrated extremity of the reference (about 0.5).

        Returns
        -------
        float
        """
        self._check_fitted()
        assert self._reference_extremity is not None
        return self._reference_extremity


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_extremity(
    scaled_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature and aggregate extremity from eosframes-scaled values.

    Parameters
    ----------
    scaled_values:
        ``(n_query, n_features)`` float array produced by the eosframes
        scaler (the output of :meth:`PreprocessPipeline.transform`).

    Returns
    -------
    per_feature:
        ``(n_query, n_features)`` extremity in ``[0, 1]``. NaN inputs stay
        NaN.
    aggregate:
        ``(n_query,)`` 66th-percentile of ``per_feature`` across features
        (see ``AGGREGATE_QUANTILE``), ignoring NaN. The shift away from
        the mean prevents the aggregate from collapsing to the
        per-feature expectation as ``n_features`` grows; downstream CDF
        calibration in :meth:`Extremity.run` further re-spreads it to
        uniform under the reference. A query whose every feature is NaN
        returns NaN.
    """
    n_query = scaled_values.shape[0]
    n_features = scaled_values.shape[1] if scaled_values.ndim > 1 else 0
    if n_features == 0:
        return np.zeros((n_query, 0)), np.full(n_query, np.nan)

    per_feature = np.minimum(np.abs(scaled_values), 1.0)
    return per_feature, _nan_aggregate(per_feature)


def _column_table(values: np.ndarray) -> np.ndarray:
    """Ascending float32 table of a column's finite reference values (may be empty)."""
    return np.sort(values[np.isfinite(values)]).astype(np.float32)


def _per_column_percentiles(
    per_feature: np.ndarray, tables: dict[str, np.ndarray]
) -> np.ndarray:
    """Mid-rank percentile of each per-feature value on its own column's table.

    Values are rounded to float32 first, as the tables are, so a reference
    value finds exactly its own rank. A column with an empty table, and NaN
    values, give NaN.

    Parameters
    ----------
    per_feature : numpy.ndarray
        ``(n, n_features)`` per-feature extremity.
    tables : dict of str to numpy.ndarray
        Ascending reference values per column, in column order.

    Returns
    -------
    numpy.ndarray
        ``(n, n_features)`` percentiles in ``(0, 1]``.
    """
    out = np.full(per_feature.shape, np.nan)
    for j, table in enumerate(tables.values()):
        if table.size:
            out[:, j] = _cdf_score(
                per_feature[:, j].astype(np.float32), table, higher_is_higher=True
            )
    return out
