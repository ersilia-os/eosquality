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
from eosquality.scores._binning import (
    binned_cdf_score,
    decode_edges,
    encode_edges,
    load_per_bin,
    min_bin_size,
    partition_and_sort,
    quantile_bin_edges,
    save_per_bin,
)
from eosquality.scores._helpers import (
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
    ) -> Consistency:
        """Fit the per-FP-distance-bin calibration on a reference DataFrame.

        Pass pre-fit ``shared`` / ``knn`` (as :class:`ErsiliaQuality` does), or
        ``eos_id`` + ``version`` so both states are fitted here.

        Parameters
        ----------
        reference : pandas.DataFrame
            Predictions on the reference library.
        vector_index : str, pathlib.Path or VectorIndex
            The reference library's vector index.
        k : int, optional
            Fingerprint neighbours.
        eos_id, version : str, optional
            Model identifier and version, needed only to fit ``shared`` here.
        shared : SharedFitState, optional
            Pre-fit shared state.
        knn : KnnFitState, optional
            Pre-fit kNN state (with its fit-time neighbour arrays).

        Returns
        -------
        Consistency
            ``self``, fitted.
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
        edges, per_bin, anchor = _fit_calibration(shared, knn, self.N_FP_BINS)
        self._shared = shared
        self._knn = knn
        self._fp_bin_edges = edges
        self._sorted_self_distances_per_bin = per_bin
        self._reference_consistency = anchor
        self._vector_index_cache = vi
        self._finish_fit(t0)
        logger.debug(
            f"Consistency fit | k={knn.k} | n_bins={self.n_bins_} | "
            f"reference_consistency={anchor:.4f}"
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
        query : pandas.DataFrame
            The reference's numeric columns plus an ``input`` SMILES column.
        query_repr : numpy.ndarray, optional
            Pre-scaled, feature-selected query array; skips validation and scaling.
        query_fp_indices, query_fp_distances : numpy.ndarray, optional
            Pre-computed ``(n_query, k)`` FP neighbours and Tanimoto distances;
            recomputed from the vector index if omitted.
        query_output_distances : numpy.ndarray, optional
            Pre-computed ``(n_query, k)`` output-space L1 distances to those
            neighbours; only valid together with the FP arrays.

        Returns
        -------
        ConsistencyRunResult
            Calibrated score, raw mean output distance and run metadata.
        """
        self._check_fitted()
        assert self._shared is not None and self._knn is not None
        if "input" not in query.columns:
            raise ValueError(
                "Consistency.run requires an 'input' column with SMILES for the vector index."
            )
        fp_distances, output_distances = self._neighbour_distances(
            query,
            query_repr,
            query_fp_indices,
            query_fp_distances,
            query_output_distances,
        )
        distance_k_mean = _mean_over_neighbors(output_distances)
        score = binned_cdf_score(
            distance_k_mean,
            fp_distances.mean(axis=1).astype(np.float64),
            self._fp_bin_edges,
            self._sorted_self_distances_per_bin,
            higher_is_higher=False,
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

    def _neighbour_distances(
        self, query, query_repr, fp_indices, fp_distances, output_distances
    ) -> tuple[np.ndarray, np.ndarray]:
        """FP and output-space distances to the k neighbours, computing what is missing."""
        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)
        if fp_indices is None or fp_distances is None:
            if output_distances is not None:
                raise ValueError(
                    "query_output_distances must be passed together with the "
                    "query_fp_indices / query_fp_distances they were computed from."
                )
            fp_distances, fp_indices = _query_fp_distances(
                query, self._get_vector_index(), self._knn.k
            )
        if output_distances is None:
            assert self._shared.ref_repr is not None
            output_distances = _query_output_distances(
                query_repr, self._shared.ref_repr, fp_indices, fp_distances
            )
        return fp_distances, output_distances

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline, bins, edges) and the per-bin CDFs."""
        assert self._fp_bin_edges is not None
        assert self._sorted_self_distances_per_bin is not None
        save_per_bin(folder / DISTANCES_FILE, self._sorted_self_distances_per_bin)
        payload = {
            "reference_consistency": self._reference_consistency,
            "n_bins": self.n_bins_,
            "fp_bin_edges": encode_edges(self._fp_bin_edges),
        }
        with open(folder / STATE_FILE, "w") as f:
            json.dump(payload, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        payload = read_json(folder / STATE_FILE, self.NAME)
        n_bins = int(payload["n_bins"])
        fp_bin_edges = decode_edges(payload["fp_bin_edges"])
        if len(fp_bin_edges) != n_bins + 1:
            raise ValueError(
                f"consistency/{STATE_FILE} declares n_bins={n_bins} but has "
                f"{len(fp_bin_edges)} bin edges."
            )
        self._sorted_self_distances_per_bin = load_per_bin(
            require_file(folder / DISTANCES_FILE, self.NAME), n_bins
        )
        self._fp_bin_edges = fp_bin_edges
        self._reference_consistency = float(payload["reference_consistency"])

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
            and self._knn is not None
            and self._fp_bin_edges is not None
            and self._sorted_self_distances_per_bin is not None
            and self._reference_consistency is not None
        )

    @property
    def n_bins_(self) -> int:
        """Number of FP-distance bins in the fitted calibration.

        Returns
        -------
        int
        """
        self._check_fitted()
        assert self._sorted_self_distances_per_bin is not None
        return len(self._sorted_self_distances_per_bin)

    @property
    def fp_bin_edges_(self) -> np.ndarray:
        """FP-distance bin edges (outer ones ``±inf``).

        Returns
        -------
        numpy.ndarray
        """
        self._check_fitted()
        assert self._fp_bin_edges is not None
        return self._fp_bin_edges

    @property
    def sorted_self_distances_per_bin_(self) -> list[np.ndarray]:
        """Per bin, the sorted reference output distances (the CDF tables).

        Returns
        -------
        list of numpy.ndarray
        """
        self._check_fitted()
        assert self._sorted_self_distances_per_bin is not None
        return self._sorted_self_distances_per_bin

    @property
    def reference_consistency_(self) -> float:
        """Mean calibrated consistency of the reference (about 0.5).

        Returns
        -------
        float
        """
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


def _fit_calibration(
    shared: SharedFitState, knn: KnnFitState, n_bins: int
) -> tuple[np.ndarray, list[np.ndarray], float]:
    """FP-distance bin edges, per-bin sorted self-distances and the reference anchor.

    Uses the same output-distance arithmetic as :meth:`Consistency.run`, so the
    fit and run paths cannot drift.
    """
    if knn.reference_knn_indices is None or knn.mean_fp_distances is None:
        raise RuntimeError(
            "Consistency.fit requires a KnnFitState that still carries the fit-time "
            "reference_knn_indices and mean_fp_distances (produced by fit_knn)."
        )
    if shared.ref_repr is None:
        raise RuntimeError("Consistency.fit needs shared.ref_repr.")
    output = _mean_over_neighbors(
        _query_output_distances(
            shared.ref_repr, shared.ref_repr, knn.reference_knn_indices
        )
    )
    fp = knn.mean_fp_distances.astype(np.float64)
    edges = quantile_bin_edges(fp, n_bins, min_bin_size=min_bin_size(len(fp), n_bins))
    per_bin = partition_and_sort(output, fp, edges)
    anchor = float(
        np.nanmean(binned_cdf_score(output, fp, edges, per_bin, higher_is_higher=False))
    )
    return edges, per_bin, anchor


def _mean_over_neighbors(distances: np.ndarray) -> np.ndarray:
    """Per-row mean over the k neighbors, ignoring NaN; all-NaN rows → NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(distances, axis=1).astype(np.float64)
