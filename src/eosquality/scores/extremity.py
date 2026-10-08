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

import pathlib
from typing import Any

import numpy as np

from eosquality.scores._base import require_file
from eosquality.scores._helpers import (
    _cdf_score,
    _nan_aggregate,
)
from eosquality.scores._percentile_score import PercentileScore

SUBFOLDER = "extremity"
COLUMN_TABLES_FILE = "column_tables.npz"


class Extremity(PercentileScore):
    """Position-based per-feature extremity scorer.

    Fitted state, beside the whole-model CDF table of
    :class:`~eosquality.scores._percentile_score.PercentileScore`:
    ``column_tables_``, per selected column the ascending reference per-feature
    values (float32), the table of each column's percentile.

    Depends only on :class:`SharedFitState`: no reference library required, no
    per-column LUTs.
    """

    NAME = SUBFOLDER
    SCORE = "ref_extremity"

    def __init__(self) -> None:
        super().__init__()
        self._column_tables: dict[str, np.ndarray] | None = None

    def _per_feature(self, scaled: np.ndarray) -> np.ndarray:
        return np.minimum(np.abs(scaled), 1.0)

    def _percentiles(self, scaled: np.ndarray, per_feature: np.ndarray) -> np.ndarray:
        assert self._column_tables is not None
        return _per_column_percentiles(per_feature, self._column_tables)

    def _fit_columns(self, scaled: np.ndarray, columns: list[str]) -> None:
        per_feature = self._per_feature(scaled)
        self._column_tables = {
            name: _column_table(per_feature[:, j]) for j, name in enumerate(columns)
        }

    def _save_columns(self, folder: pathlib.Path) -> dict[str, Any]:
        assert self._column_tables is not None
        np.savez(
            folder / COLUMN_TABLES_FILE,
            **{f"c{j:03d}": t for j, t in enumerate(self._column_tables.values())},
        )
        return {}

    def _load_columns(self, folder: pathlib.Path, state: dict[str, Any]) -> None:
        with np.load(require_file(folder / COLUMN_TABLES_FILE, self.NAME)) as npz:
            self._column_tables = {
                name: npz[f"c{j:03d}"] for j, name in enumerate(state["columns"])
            }

    def _has_columns(self) -> bool:
        return self._column_tables is not None

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
            out[:, j] = _cdf_score(per_feature[:, j].astype(np.float32), table)
    return out
