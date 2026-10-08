"""Typicality score: per-feature + CDF-calibrated aggregate, no library needed.

Typicality is **density-based**: for each query value, look up the
per-column count LUT built on the reference (the reference's counts per int8
level) and return ``count / max_count``. The lookup interpolates linearly
between the two int8 levels around the value rather than rounding to one, so
the density varies continuously with the value instead of in steps.

This handles every distribution shape uniformly — unimodal, multimodal,
constant, binary — with no kind dispatch: the most common level scores
typicality 1.0, every other level scores in proportion to how often it appears
in the reference, and unseen levels score 0.

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
  sorted distribution of it (:func:`_cdf_score`). The result is
  uniform under the reference and comparable across models with different
  feature counts. The per-feature raw values and percentiles are returned as
  ``per_feature`` and ``per_feature_pct``.

A column's percentile table is the mid-rank of the density among the
reference's own densities, kept as a 4096-bin histogram of them
(:func:`density_histograms`, saved); the table is derived from it
(:func:`percentile_tables`). Densities that are exactly equal (a binary or
constant column) fall in one bin and tie at half weight, as before.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from eosquality.scores._base import require_file
from eosquality.scores._helpers import _nan_aggregate
from eosquality.scores._percentile_score import PercentileScore

_INT8_MAX_VAL = 127
_LUT_SIZE = 256
_LUT_OFFSET = 128  # lut index = int8 + offset; the slot at index 0 is the NaN sentinel
_N_BINS = 4096  # bins of the density histogram behind the percentile tables
SUBFOLDER = "typicality"
COUNT_LUTS_FILE = "count_luts.npy"
DENSITY_HIST_FILE = "density_hist.npy"


class Typicality(PercentileScore):
    """Density-based per-feature typicality scorer.

    Fitted state, beside the whole-model CDF table of
    :class:`~eosquality.scores._percentile_score.PercentileScore`:

    - per-column int8 count LUTs, ``(256, n_features)``: reference counts per
      level per column, built at fit time (saved);
    - per-column histogram of the reference's densities, ``(4096, n_features)``
      (saved);
    - ``pct_tables_`` — ``(4096, n_features)`` per-column percentile of each
      density bin, derived from the histogram (not saved).

    Depends only on :class:`SharedFitState` — no reference library required.
    """

    NAME = SUBFOLDER
    SCORE = "ref_typicality"

    def __init__(self) -> None:
        super().__init__()
        self._count_luts: np.ndarray | None = None  # (256, n_features)
        self._density_hist: np.ndarray | None = None  # (4096, n_features)
        self._pct_tables: np.ndarray | None = None  # (4096, n_features), derived

    def _per_feature(self, scaled: np.ndarray) -> np.ndarray:
        assert self._count_luts is not None
        return compute_typicality(scaled, self._count_luts)[0]

    def _percentiles(self, scaled: np.ndarray, per_feature: np.ndarray) -> np.ndarray:
        assert self._pct_tables is not None
        return lookup_percentiles(per_feature, self._pct_tables)

    def _fit_columns(self, scaled: np.ndarray, columns: list[str]) -> None:
        self._count_luts = fit_typicality_luts(scaled)
        self._density_hist = density_histograms(scaled, self._count_luts)
        self._pct_tables = percentile_tables(self._density_hist)

    def _save_columns(self, folder: pathlib.Path) -> dict[str, Any]:
        assert self._count_luts is not None and self._density_hist is not None
        np.save(folder / COUNT_LUTS_FILE, self._count_luts)
        np.save(folder / DENSITY_HIST_FILE, self._density_hist)
        return {}

    def _load_columns(self, folder: pathlib.Path, state: dict[str, Any]) -> None:
        self._count_luts = np.load(require_file(folder / COUNT_LUTS_FILE, self.NAME))
        self._density_hist = np.load(
            require_file(folder / DENSITY_HIST_FILE, self.NAME)
        )
        self._pct_tables = percentile_tables(self._density_hist)

    def _has_columns(self) -> bool:
        return self._count_luts is not None and self._pct_tables is not None

    @property
    def pct_tables_(self) -> np.ndarray:
        """Per-column percentile of each density bin (derived from the histogram).

        Returns
        -------
        numpy.ndarray
            ``(4096, n_features)``.
        """
        self._check_fitted()
        assert self._pct_tables is not None
        return self._pct_tables


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


def _interpolate(luts: np.ndarray, scaled: np.ndarray) -> np.ndarray:
    """Per-column LUT value at scaled values, linear between int8 levels.

    A value ``x`` sits at ``t = x × 127`` on the level axis (a level's centre is
    its integer); the result is the LUT value of the two levels around ``t``,
    weighted by distance. A value on a level gives that level's entry. ``t``
    is clipped to ``[-127, 127]``, and NaN stays NaN.
    """
    t = np.clip(scaled * _INT8_MAX_VAL, -_INT8_MAX_VAL, _INT8_MAX_VAL)
    low = np.floor(np.nan_to_num(t))
    weight = t - low
    low = low.astype(np.int64)
    high = np.minimum(low + 1, _INT8_MAX_VAL)
    cols = np.arange(scaled.shape[1])[None, :]
    return (1.0 - weight) * luts[low + _LUT_OFFSET, cols] + weight * luts[
        high + _LUT_OFFSET, cols
    ]


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

    col_max = count_luts.max(axis=0).astype(np.float64)
    col_max[col_max <= 0] = 1.0
    per_feature = _interpolate(count_luts, scaled_values) / col_max[None, :]
    return per_feature, _nan_aggregate(per_feature)


def density_histograms(
    scaled_reference: np.ndarray, count_luts: np.ndarray
) -> np.ndarray:
    """Histogram of the reference's own densities, per column.

    Parameters
    ----------
    scaled_reference : numpy.ndarray
        ``(n_ref, n_features)`` eosframes-scaled reference values.
    count_luts : numpy.ndarray
        ``(256, n_features)`` counts, as returned by :func:`fit_typicality_luts`.

    Returns
    -------
    numpy.ndarray
        ``(4096, n_features)`` int64: how many reference values have a
        density (``count / max_count``, interpolated) in each of 4096 equal
        bins of ``[0, 1]``. NaN values are not counted.
    """
    n_features = scaled_reference.shape[1]
    hist = np.zeros((_N_BINS, n_features), dtype=np.int64)
    for j in range(n_features):
        density = compute_typicality(scaled_reference[:, [j]], count_luts[:, [j]])[0]
        valid = np.isfinite(density[:, 0])
        hist[:, j] = np.bincount(_density_bin(density[valid, 0]), minlength=_N_BINS)
    return hist


def percentile_tables(density_hist: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each density bin among the reference's densities.

    The share of reference values whose density is in a lower bin plus half
    the share in the same bin, as in
    :func:`~eosquality.scores._helpers._cdf_score`: the densest values score
    near 1, rare ones near 0. A column with no counted value gives NaN.

    Parameters
    ----------
    density_hist : numpy.ndarray
        ``(4096, n_features)`` counts, as returned by :func:`density_histograms`.

    Returns
    -------
    numpy.ndarray
        ``(4096, n_features)`` percentiles in ``(0, 1]``.
    """
    counts = np.asarray(density_hist, dtype=np.float64)
    n = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = (np.cumsum(counts, axis=0) - 0.5 * counts) / n
        return np.where(n > 0, np.clip(mid, 0.5 / n, 1.0), np.nan)


def lookup_percentiles(per_feature: np.ndarray, pct_tables: np.ndarray) -> np.ndarray:
    """Per-feature typicality percentiles of per-feature densities.

    Parameters
    ----------
    per_feature : numpy.ndarray
        ``(n, n_features)`` densities from :func:`compute_typicality`.
    pct_tables : numpy.ndarray
        ``(4096, n_features)`` from :func:`percentile_tables`.

    Returns
    -------
    numpy.ndarray
        ``(n, n_features)`` percentiles; NaN where the density is NaN.
    """
    if per_feature.shape[1] == 0:
        return np.ones((per_feature.shape[0], 0))
    bins = _density_bin(np.nan_to_num(per_feature))
    out = pct_tables[bins, np.arange(per_feature.shape[1])[None, :]]
    out[np.isnan(per_feature)] = np.nan
    return out


def _density_bin(density: np.ndarray) -> np.ndarray:
    """Histogram bin of densities in ``[0, 1]``."""
    return np.minimum((density * _N_BINS).astype(np.int64), _N_BINS - 1)
