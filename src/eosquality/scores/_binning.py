"""Conditional (binned) CDF calibration, used by Consistency.

The raw value is calibrated against the reference *within a stratum* of a
conditioning variable (for Consistency, the mean FP distance to the k
neighbours). The reference is split into quantile bins of
the conditioning key; each bin keeps its own sorted reference values; a
query is scored against the bin its own key falls in. Because each bin is
calibrated on itself, reference rows still score ~Uniform(0, 1) overall.
"""

from __future__ import annotations

import pathlib

import numpy as np

from eosquality.scores._helpers import _cdf_score

# A bin's CDF needs enough rows to resolve scores: at least this many, or
# half the nominal bin size for small references.
MIN_BIN_ROWS = 1_000

# ±inf bin edges are stored as these sentinels so the JSON stays plain.
_NEG_INF_SENTINEL = -1.0e18
_POS_INF_SENTINEL = 1.0e18


def min_bin_size(n_reference: int, n_bins: int) -> int:
    return max(1, min(MIN_BIN_ROWS, n_reference // (2 * n_bins)))


def quantile_bin_edges(
    keys: np.ndarray, n_bins: int, *, min_bin_size: int = 1
) -> np.ndarray:
    """Return ascending bin edges ``[-inf, e_1, …, e_m, +inf]`` over ``keys``.

    Starts from ``n_bins`` quantile bins, then (1) collapses duplicate
    interior quantiles (heavy ties, e.g. integer keys) and (2) repeatedly
    merges the smallest bin into its smaller neighbor until every bin holds
    at least ``min_bin_size`` rows. A key equal to an edge falls in the bin
    above it (``side="right"``, as in :func:`assign_bins`). Outer edges are
    ``±inf`` so any query key lands in a bin. The result can have fewer
    than ``n_bins`` bins.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    keys = np.asarray(keys, dtype=np.float64)
    interior_q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = np.unique(np.quantile(keys, interior_q))
    while interior.size:
        edges = np.concatenate(([-np.inf], interior, [np.inf]))
        counts = np.bincount(assign_bins(keys, edges), minlength=edges.size - 1)
        smallest = int(np.argmin(counts))
        if counts[smallest] >= min_bin_size:
            break
        # Interior edge i separates bins i and i+1: drop the one shared
        # with the smaller neighbor.
        if smallest == 0:
            drop = 0
        elif smallest == counts.size - 1:
            drop = smallest - 1
        else:
            left_smaller = counts[smallest - 1] <= counts[smallest + 1]
            drop = smallest - 1 if left_smaller else smallest
        interior = np.delete(interior, drop)
    return np.concatenate(([-np.inf], interior.astype(np.float64), [np.inf]))


def assign_bins(keys: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-row bin index (``0 … n_bins-1``) of each key."""
    n_bins = len(edges) - 1
    bin_idx = np.searchsorted(edges[1:-1], keys, side="right")
    return np.clip(bin_idx, 0, n_bins - 1).astype(np.int64)


def partition_and_sort(
    values: np.ndarray, keys: np.ndarray, edges: np.ndarray
) -> list[np.ndarray]:
    """Per bin of ``keys``, the ascending finite ``values`` (one CDF per bin)."""
    bin_idx = assign_bins(keys, edges)
    finite = np.isfinite(values)
    return [
        np.sort(values[(bin_idx == b) & finite]).astype(np.float64)
        for b in range(len(edges) - 1)
    ]


def binned_cdf_score(
    values: np.ndarray,
    keys: np.ndarray,
    edges: np.ndarray,
    sorted_per_bin: list[np.ndarray],
    *,
    higher_is_higher: bool,
) -> np.ndarray:
    """Score each value with :func:`_cdf_score` against its key's bin.

    NaN values score NaN, as do rows routed to a bin with no finite
    reference value (only possible if the reference is almost all missing).
    """
    bin_idx = assign_bins(keys, edges)
    score = np.full(np.shape(values), np.nan, dtype=np.float64)
    for b, sorted_arr in enumerate(sorted_per_bin):
        mask = bin_idx == b
        if mask.any() and sorted_arr.size:
            score[mask] = _cdf_score(
                values[mask], sorted_arr, higher_is_higher=higher_is_higher
            )
    return score


def encode_edges(edges: np.ndarray) -> list[float]:
    """JSON-safe edges: ``±inf`` → ``±1e18``."""
    return [
        (
            _NEG_INF_SENTINEL
            if v == -np.inf
            else _POS_INF_SENTINEL
            if v == np.inf
            else float(v)
        )
        for v in edges
    ]


def decode_edges(encoded: list[float]) -> np.ndarray:
    """Inverse of :func:`encode_edges`."""
    return np.asarray(
        [
            (
                -np.inf
                if v <= _NEG_INF_SENTINEL
                else np.inf
                if v >= _POS_INF_SENTINEL
                else v
            )
            for v in encoded
        ],
        dtype=np.float64,
    )


def save_per_bin(path: pathlib.Path, sorted_per_bin: list[np.ndarray]) -> None:
    np.savez(path, **{f"b{i:02d}": arr for i, arr in enumerate(sorted_per_bin)})


def load_per_bin(path: pathlib.Path, n_bins: int) -> list[np.ndarray]:
    with np.load(path) as npz:
        return [np.asarray(npz[f"b{i:02d}"], dtype=np.float64) for i in range(n_bins)]
