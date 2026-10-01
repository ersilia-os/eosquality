"""Consistency score: where the query's output-space neighborhood noise sits
in the reference's self-noise CDF, *conditioned on the FP-distance regime*.

Per-query distance is the mean output-space L1 distance from the query to its
``k`` FP-nearest reference neighbors. The score then maps that distance
through the reference's self-output-distance CDF — **but the CDF is
conditional on the query's FP-distance bin**. This decouples consistency
from support: a query whose neighborhood is far in FP space is scored
against reference rows whose neighborhood is also far, so "is my prediction
surprising for the FP-distance regime I'm in?" replaces the old
"is my prediction surprising vs. a globally in-distribution null?".

The reference is partitioned into (up to) ``N_FP_BINS`` quantile bins on
``knn.mean_fp_distances``; duplicate quantile edges are merged and bins
smaller than a minimum size are merged into a neighbor. Each bin gets its
own sorted ``self_output_distance`` CDF. At run time, queries are routed to
the bin whose FP-distance interval contains their own mean FP-distance.

Output-space distances average |difference| over the features that are
finite on both sides; a query with no usable feature scores NaN.

Closer than every reference point *in the same bin* → ~1.0; at that bin's
median → ~0.5; farther than every reference point in that bin → eps.

Operates in **output space**. Support is the FP-space sibling that does
unconditional CDF calibration on Tanimoto distances.
"""

from __future__ import annotations

import json
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.knn.state import KnnFitState
from eosquality.schema.infer import validate_against_schema
from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import (
    _cdf_score,
    _make_query_repr,
    _query_fp_distances,
    _query_output_distances,
    _resolve_shared_and_knn,
    _resolve_vector_index,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

SUBFOLDER = "consistency"
STATE_FILE = "state.json"
DISTANCES_FILE = "reference_self_distances_per_bin.npz"

# Sentinel encoding of ``±inf`` in JSON, so the JSON parser doesn't have to
# deal with ``Infinity`` literals.
_NEG_INF_SENTINEL = -1.0e18
_POS_INF_SENTINEL = 1.0e18


@dataclass
class ConsistencyRunResult:
    """Result returned by :meth:`Consistency.run`."""

    score: pd.Series  # (n_query,) calibrated consistency in (0, 1]
    score_raw: pd.Series  # (n_query,) raw mean output-space L1 (= distance_k_mean)
    distance_k_mean: pd.Series  # mean output-space L1 to k neighbors
    metadata: dict[str, Any] = field(default_factory=dict)


class Consistency(ScoreComponent):
    """CDF-based output-space neighborhood-noise scorer (conditional on FP regime).

    Holds four pieces of fitted state on top of the shared and kNN states:

    - ``fp_bin_edges_`` — ``(n_bins + 1,)`` ascending bin edges over
      reference mean FP distance. Outer edges are ``±inf`` so any query
      FP-distance maps cleanly into one of the bins.
    - ``sorted_self_distances_per_bin_`` — list of length ``n_bins``;
      each entry is the sorted reference mean output-space L1 distances
      for rows whose own mean FP distance falls in that bin. The
      conditional CDF lookup table.
    - ``reference_consistency_`` — mean reference-as-query consistency
      under those CDFs. A calibration anchor; ≈ 0.5 for a healthy
      reference because each bin is calibrated on itself.
    - ``n_bins_`` — number of FP-distance bins actually used.

    Reads ``shared.ref_repr`` at run time to compute output-space neighbor
    distances. Mirrors :class:`Support` in shape, but support uses an
    unconditional CDF on FP distance and consistency uses a
    conditional-on-FP-distance CDF on output distance.
    """

    NAME = SUBFOLDER
    USES_KNN = True
    N_FP_BINS = 10  # target number of quantile bins on reference mean FP distance

    def __init__(self) -> None:
        super().__init__()
        self._fp_bin_edges: np.ndarray | None = None
        self._sorted_self_distances_per_bin: list[np.ndarray] | None = None
        self._reference_consistency: float | None = None
        self._vector_index_cache: VectorIndex | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        *,
        vector_index: str | pathlib.Path | VectorIndex,
        k: int = 5,
        eos_id: str | None = None,
        version: str | None = None,
        shared: SharedFitState | None = None,
        knn: KnnFitState | None = None,
    ) -> "Consistency":
        """Fit on a reference DataFrame.

        Computes the reference's per-row mean output-space L1 distance to
        its k FP-selected neighbors, partitions rows into FP-distance
        quantile bins, and sorts each bin's distances to form the
        conditional CDF lookup tables.

        Either pass pre-fit ``shared=`` / ``knn=`` (when composed by
        :class:`ErsiliaQuality`), or pass ``eos_id`` + ``version`` +
        ``vector_index`` so Consistency can fit both states itself.
        """
        t0 = time.perf_counter()
        shared, knn, vi = _resolve_shared_and_knn(
            reference=reference,
            vector_index=vector_index,
            k=k,
            eos_id=eos_id,
            version=version,
            shared=shared,
            knn=knn,
        )
        if knn.reference_knn_indices is None or knn.mean_fp_distances is None:
            raise RuntimeError(
                "Consistency.fit requires a KnnFitState that still carries the "
                "fit-time reference_knn_indices and mean_fp_distances (i.e., "
                "produced by fit_knn in this pass)."
            )
        if shared.ref_repr is None:
            raise RuntimeError(
                "Consistency.fit needs shared.ref_repr to compute the "
                "reference's own output-space self-distances."
            )

        # Output-space self-kNN distances — same arithmetic as run() so the
        # two paths cannot drift.
        mean_self_output_distances = _mean_over_neighbors(
            _query_output_distances(
                shared.ref_repr, shared.ref_repr, knn.reference_knn_indices
            )
        )
        mean_self_fp_distances = knn.mean_fp_distances.astype(np.float64)

        fp_bin_edges = _compute_fp_bin_edges(
            mean_self_fp_distances,
            self.N_FP_BINS,
            min_bin_size=_min_bin_size(len(mean_self_fp_distances), self.N_FP_BINS),
        )
        sorted_self_distances_per_bin = _partition_and_sort(
            values=mean_self_output_distances,
            keys=mean_self_fp_distances,
            bin_edges=fp_bin_edges,
        )

        self._shared = shared
        self._knn = knn
        self._fp_bin_edges = fp_bin_edges
        self._sorted_self_distances_per_bin = sorted_self_distances_per_bin
        self._reference_consistency = float(
            np.nanmean(
                _consistency_from_distances_binned(
                    distance_k_mean=mean_self_output_distances,
                    fp_distance_k_mean=mean_self_fp_distances,
                    fp_bin_edges=fp_bin_edges,
                    sorted_self_distances_per_bin=sorted_self_distances_per_bin,
                )
            )
        )
        self._vector_index_cache = vi
        self._finish_fit(t0)
        logger.debug(
            f"Consistency fit | k={knn.k} | n_ref={len(shared.reference_ids):,}"
            f" | n_bins={self.n_bins_}"
            f" | reference_consistency={self._reference_consistency:.4f}"
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
        query_fp_indices: np.ndarray | None = None,
        query_fp_distances: np.ndarray | None = None,
        query_output_distances: np.ndarray | None = None,
    ) -> ConsistencyRunResult:
        """Score query samples.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference plus
            an ``'input'`` SMILES column for the vector index.
        query_repr:
            Optional pre-scaled, feature-selected query array
            ``(n_query, n_selected)``. If provided, schema validation and
            the eosframes transform are skipped.
        query_fp_indices, query_fp_distances:
            Optional pre-computed FP-selected neighbor indices and Tanimoto
            distances, each ``(n_query, k)``. The distances route each query
            into its FP-distance bin. If omitted, both are recomputed from
            the vector index.
        query_output_distances:
            Optional pre-computed output-space L1 distances to those same
            neighbors ``(n_query, k)``. Only valid together with
            ``query_fp_indices`` / ``query_fp_distances``.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._knn is not None
        assert self._fp_bin_edges is not None
        assert self._sorted_self_distances_per_bin is not None

        if "input" not in query.columns:
            raise ValueError(
                "Consistency.run requires an 'input' column with SMILES for the vector index."
            )
        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)

        if query_fp_indices is None or query_fp_distances is None:
            if query_output_distances is not None:
                raise ValueError(
                    "query_output_distances must be passed together with the "
                    "query_fp_indices / query_fp_distances they were computed from."
                )
            query_fp_distances, query_fp_indices = _query_fp_distances(
                query, self._get_vector_index(), self._knn.k
            )
        if query_output_distances is None:
            assert self._shared.ref_repr is not None
            query_output_distances = _query_output_distances(
                query_repr, self._shared.ref_repr, query_fp_indices
            )

        distance_k_mean = _mean_over_neighbors(query_output_distances)
        score = _consistency_from_distances_binned(
            distance_k_mean=distance_k_mean,
            fp_distance_k_mean=query_fp_distances.mean(axis=1).astype(np.float64),
            fp_bin_edges=self._fp_bin_edges,
            sorted_self_distances_per_bin=self._sorted_self_distances_per_bin,
        )

        idx = list(query.index)
        return ConsistencyRunResult(
            score=pd.Series(score, index=idx, name="consistency"),
            score_raw=pd.Series(distance_k_mean, index=idx, name="consistency_raw"),
            distance_k_mean=pd.Series(
                distance_k_mean, index=idx, name="distance_k_mean"
            ),
            metadata={
                "reference_consistency": self._reference_consistency,
                "n_reference": len(self._shared.reference_ids),
                "k": int(self._knn.k),
                "n_fp_bins": self.n_bins_,
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline, bins, edges) and the per-bin CDFs."""
        assert self._fp_bin_edges is not None
        assert self._sorted_self_distances_per_bin is not None
        np.savez(
            folder / DISTANCES_FILE,
            **{
                f"b{i:02d}": arr
                for i, arr in enumerate(self._sorted_self_distances_per_bin)
            },
        )
        payload = {
            "reference_consistency": self._reference_consistency,
            "n_bins": self.n_bins_,
            "fp_bin_edges": _encode_finite_edges(self._fp_bin_edges),
        }
        with open(folder / STATE_FILE, "w") as f:
            json.dump(payload, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        payload = read_json(folder / STATE_FILE, self.NAME)
        n_bins = int(payload["n_bins"])
        fp_bin_edges = _decode_finite_edges(payload["fp_bin_edges"])
        if len(fp_bin_edges) != n_bins + 1:
            raise ValueError(
                f"consistency/{STATE_FILE} declares n_bins={n_bins} but has "
                f"{len(fp_bin_edges)} bin edges."
            )
        with np.load(require_file(folder / DISTANCES_FILE, self.NAME)) as npz:
            self._sorted_self_distances_per_bin = [
                np.asarray(npz[f"b{i:02d}"], dtype=np.float64) for i in range(n_bins)
            ]
        self._fp_bin_edges = fp_bin_edges
        self._reference_consistency = float(payload["reference_consistency"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._knn is not None
            and self._fp_bin_edges is not None
            and self._sorted_self_distances_per_bin is not None
            and self._reference_consistency is not None
        )

    @property
    def n_bins_(self) -> int:
        """Number of FP-distance bins in the fitted calibration."""
        self._check_fitted()
        assert self._sorted_self_distances_per_bin is not None
        return len(self._sorted_self_distances_per_bin)

    @property
    def fp_bin_edges_(self) -> np.ndarray:
        self._check_fitted()
        assert self._fp_bin_edges is not None
        return self._fp_bin_edges

    @property
    def sorted_self_distances_per_bin_(self) -> list[np.ndarray]:
        self._check_fitted()
        assert self._sorted_self_distances_per_bin is not None
        return self._sorted_self_distances_per_bin

    @property
    def reference_consistency_(self) -> float:
        self._check_fitted()
        assert self._reference_consistency is not None
        return self._reference_consistency

    def _get_vector_index(self) -> VectorIndex:
        assert self._shared is not None
        if self._vector_index_cache is None:
            self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache


# ---------------------------------------------------------------------------
# Consistency-specific helpers
# ---------------------------------------------------------------------------


# A bin's CDF needs enough rows to resolve scores: at least this many, or
# half the nominal bin size for small references.
_MIN_BIN_ROWS = 1_000


def _min_bin_size(n_reference: int, n_bins: int) -> int:
    return max(1, min(_MIN_BIN_ROWS, n_reference // (2 * n_bins)))


def _mean_over_neighbors(distances: np.ndarray) -> np.ndarray:
    """Per-row mean over the k neighbors, ignoring NaN; all-NaN rows → NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(distances, axis=1).astype(np.float64)


def _compute_fp_bin_edges(
    mean_fp_distances: np.ndarray, n_bins: int, *, min_bin_size: int = 1
) -> np.ndarray:
    """Return ascending FP-distance bin edges ``[-inf, e_1, …, e_m, +inf]``.

    Starts from the ``n_bins`` quantile bins, then (1) collapses duplicate
    interior quantiles (heavy ties in the reference's mean FP distance) and
    (2) repeatedly merges the smallest bin into its smaller neighbor until
    every bin holds at least ``min_bin_size`` reference rows. Rows equal to
    an edge fall in the bin above it (same ``side="right"`` routing as
    :func:`_assign_fp_bins`). Outer edges are ``±inf`` so any query distance
    lands in a bin. The result can therefore have fewer than ``n_bins``
    bins; :attr:`Consistency.n_bins_` reports the actual count.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    interior_q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = np.unique(np.quantile(mean_fp_distances, interior_q))
    while interior.size:
        edges = np.concatenate(([-np.inf], interior, [np.inf]))
        counts = np.bincount(
            _assign_fp_bins(mean_fp_distances, edges), minlength=edges.size - 1
        )
        smallest = int(np.argmin(counts))
        if counts[smallest] >= min_bin_size:
            break
        # Remove the edge shared with the smaller neighbor (interior edge
        # i separates bins i and i+1).
        if smallest == 0:
            drop = 0
        elif smallest == counts.size - 1:
            drop = smallest - 1
        else:
            drop = (
                smallest - 1
                if counts[smallest - 1] <= counts[smallest + 1]
                else smallest
            )
        interior = np.delete(interior, drop)
    return np.concatenate(([-np.inf], interior.astype(np.float64), [np.inf]))


def _assign_fp_bins(fp_distances: np.ndarray, fp_bin_edges: np.ndarray) -> np.ndarray:
    """Return per-row bin index (0..n_bins-1) for each FP distance value."""
    n_bins = len(fp_bin_edges) - 1
    bin_idx = np.searchsorted(fp_bin_edges[1:-1], fp_distances, side="right")
    return np.clip(bin_idx, 0, n_bins - 1).astype(np.int64)


def _partition_and_sort(
    values: np.ndarray,
    keys: np.ndarray,
    bin_edges: np.ndarray,
) -> list[np.ndarray]:
    """Per bin of ``keys``, the ascending finite ``values`` (one CDF per bin)."""
    n_bins = len(bin_edges) - 1
    bin_idx = _assign_fp_bins(keys, bin_edges)
    finite = np.isfinite(values)
    return [
        np.sort(values[(bin_idx == b) & finite]).astype(np.float64)
        for b in range(n_bins)
    ]


def _consistency_from_distances_binned(
    distance_k_mean: np.ndarray,
    fp_distance_k_mean: np.ndarray,
    fp_bin_edges: np.ndarray,
    sorted_self_distances_per_bin: list[np.ndarray],
) -> np.ndarray:
    """Map per-row output-space k-distances to consistency via per-bin CDFs.

    Each row is routed to its FP-distance bin and scored with
    :func:`_cdf_score` (distance direction) against that bin's sorted
    reference distances. NaN distances score NaN, as do rows routed to a bin
    with no finite reference distance (only possible if the reference
    outputs are almost entirely missing).
    """
    bin_idx = _assign_fp_bins(fp_distance_k_mean, fp_bin_edges)
    score = np.full(distance_k_mean.shape, np.nan, dtype=np.float64)
    for b, sorted_arr in enumerate(sorted_self_distances_per_bin):
        mask = bin_idx == b
        if mask.any() and sorted_arr.size:
            score[mask] = _cdf_score(
                distance_k_mean[mask], sorted_arr, higher_is_higher=False
            )
    return score


def _encode_finite_edges(edges: np.ndarray) -> list[float]:
    """Encode ``±inf`` as ``±_POS_INF_SENTINEL`` so JSON serialization is clean."""
    out: list[float] = []
    for v in edges:
        if v == -np.inf:
            out.append(_NEG_INF_SENTINEL)
        elif v == np.inf:
            out.append(_POS_INF_SENTINEL)
        else:
            out.append(float(v))
    return out


def _decode_finite_edges(encoded: list[float]) -> np.ndarray:
    """Inverse of :func:`_encode_finite_edges`."""
    out = np.empty(len(encoded), dtype=np.float64)
    for i, v in enumerate(encoded):
        if v <= _NEG_INF_SENTINEL:
            out[i] = -np.inf
        elif v >= _POS_INF_SENTINEL:
            out[i] = np.inf
        else:
            out[i] = float(v)
    return out
