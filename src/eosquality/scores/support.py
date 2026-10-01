"""Support score: how close the query is to the reference, for its fingerprint size.

The raw value is the mean Tanimoto distance from the query to its k
FP-nearest reference molecules (``support_raw``). Tanimoto distance on
Morgan bits is size-biased — molecules with few set bits sit farther from
everything — so the calibration is **conditioned on fingerprint size**: the
reference is split into quantile bins of its set-bit count, each bin keeps
its own CDF of self-distances, and a query is scored against the bin of its
own size. Within each bin (and therefore overall) reference molecules score
~Uniform(0, 1): closer than every same-size reference molecule → ~1.0, at
the median → ~0.5, farther than all → eps.

``support_log = −log10(support)`` re-expresses the same tail probability on
a log scale, so queries far outside the reference (where ``support`` is
squeezed into 0–0.01) remain distinguishable: ~0.3 for a typical reference
molecule, 2 means "farther than 99% of same-size reference molecules".

Operates in fingerprint space only. Consistency is the sibling that lives
in output space.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.knn.state import KnnFitState
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
    _query_fp_distances,
    _resolve_shared_and_knn,
    _resolve_vector_index,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

SUBFOLDER = "support"
STATE_FILE = "state.json"
DISTANCES_FILE = "reference_self_distances_per_bin.npz"


@dataclass
class SupportRunResult:
    """Result returned by :meth:`Support.run`."""

    score: pd.Series  # (n_query,) calibrated support in (0, 1]
    score_raw: pd.Series  # (n_query,) raw mean FP Tanimoto distance (= distance_k_mean)
    score_log: pd.Series  # (n_query,) −log10(support), ≥ 0
    distance_k_mean: pd.Series  # mean FP (Tanimoto) distance to k neighbors
    distance_k_max: pd.Series  # max FP (Tanimoto) distance to k neighbors
    nearest_reference_ids: list[list[Any]]
    fingerprint_size: pd.Series  # set Morgan bits of the query (the conditioning key)
    metadata: dict[str, Any] = field(default_factory=dict)


class Support(ScoreComponent):
    """Size-conditioned CDF support scorer (FP-space Tanimoto distances).

    Fitted state on top of the shared and kNN states:

    - ``size_bin_edges_`` — ascending edges over the reference's set-bit
      counts (outer edges ``±inf``), up to ``N_SIZE_BINS`` bins after
      merging ties and small bins.
    - ``sorted_self_distances_per_bin_`` — per bin, the sorted mean FP
      k-distances of the reference molecules of that size.
    - ``reference_support_`` — mean reference-as-query support (≈ 0.5).

    Loads the underlying :class:`VectorIndex` lazily on first :meth:`run`.
    """

    NAME = SUBFOLDER
    USES_KNN = True
    N_SIZE_BINS = 10  # target number of quantile bins on fingerprint size

    def __init__(self) -> None:
        super().__init__()
        self._size_bin_edges: np.ndarray | None = None
        self._sorted_self_distances_per_bin: list[np.ndarray] | None = None
        self._reference_support: float | None = None
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
    ) -> Support:
        """Fit on a reference DataFrame.

        Reads the reference's FP self-kNN Tanimoto distances (identity
        neighbor already stripped at index build time) and the set-bit count
        of every reference fingerprint, bins the reference by size, and sorts
        each bin's mean FP distances into its own CDF.

        Either pass pre-fit ``shared=`` / ``knn=`` (when composed by
        :class:`ErsiliaQuality`), or pass ``eos_id`` + ``version`` +
        ``vector_index`` so Support can fit both states itself.
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
        if knn.mean_fp_distances is None:
            raise RuntimeError(
                "Support.fit requires a KnnFitState that still carries fit-time "
                "mean_fp_distances (i.e., produced by fit_knn in this pass)."
            )
        distances = knn.mean_fp_distances.astype(np.float64)
        sizes = vi.fingerprint_sizes().astype(np.float64)
        edges = quantile_bin_edges(
            sizes,
            self.N_SIZE_BINS,
            min_bin_size=min_bin_size(len(sizes), self.N_SIZE_BINS),
        )
        per_bin = partition_and_sort(distances, sizes, edges)

        self._shared = shared
        self._knn = knn
        self._size_bin_edges = edges
        self._sorted_self_distances_per_bin = per_bin
        self._reference_support = float(
            np.nanmean(
                binned_cdf_score(
                    distances, sizes, edges, per_bin, higher_is_higher=False
                )
            )
        )
        self._vector_index_cache = vi
        self._finish_fit(t0)
        logger.debug(
            f"Support fit | k={knn.k} | n_ref={len(distances):,} | "
            f"size bins={len(per_bin)} | reference_support={self._reference_support:.4f}"
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
        query_fp_indices: np.ndarray | None = None,
        query_fp_distances: np.ndarray | None = None,
    ) -> SupportRunResult:
        """Score query samples.

        Parameters
        ----------
        query:
            DataFrame with an ``'input'`` SMILES column for the vector
            index (other columns are ignored — Support is FP-only).
        query_fp_indices, query_fp_distances:
            Optional pre-computed FP-selected neighbor indices and their
            Tanimoto distances, each ``(n_query, k)``. Must be passed
            together — used by :class:`ErsiliaQuality` to share the FP
            query across scores.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._knn is not None
        assert self._size_bin_edges is not None
        assert self._sorted_self_distances_per_bin is not None

        if "input" not in query.columns:
            raise ValueError(
                "Support.run requires an 'input' column with SMILES for the vector index."
            )
        vi = self._get_vector_index()
        if query_fp_indices is None or query_fp_distances is None:
            query_fp_distances, query_fp_indices = _query_fp_distances(
                query, vi, self._knn.k
            )

        distance_k_mean = query_fp_distances.mean(axis=1)
        sizes = vi.query_fingerprint_sizes(list(query["input"]))
        support_score = binned_cdf_score(
            distance_k_mean,
            sizes,
            self._size_bin_edges,
            self._sorted_self_distances_per_bin,
            higher_is_higher=False,
        )

        idx = list(query.index)
        reference_ids = self._shared.reference_ids
        return SupportRunResult(
            score=pd.Series(support_score, index=idx, name="support"),
            score_raw=pd.Series(distance_k_mean, index=idx, name="support_raw"),
            score_log=pd.Series(
                -np.log10(support_score), index=idx, name="support_log"
            ),
            distance_k_mean=pd.Series(
                distance_k_mean, index=idx, name="distance_k_mean"
            ),
            distance_k_max=pd.Series(
                query_fp_distances.max(axis=1), index=idx, name="distance_k_max"
            ),
            nearest_reference_ids=[
                [reference_ids[j] for j in row] for row in query_fp_indices
            ],
            fingerprint_size=pd.Series(sizes, index=idx, name="fingerprint_size"),
            metadata={
                "reference_support": self._reference_support,
                "n_reference": len(reference_ids),
                "k": int(self._knn.k),
                "n_size_bins": self.n_bins_,
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline, size bins) and the per-bin CDFs."""
        assert self._size_bin_edges is not None
        assert self._sorted_self_distances_per_bin is not None
        save_per_bin(folder / DISTANCES_FILE, self._sorted_self_distances_per_bin)
        payload = {
            "reference_support": self._reference_support,
            "n_bins": self.n_bins_,
            "size_bin_edges": encode_edges(self._size_bin_edges),
        }
        with open(folder / STATE_FILE, "w") as f:
            json.dump(payload, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        payload = read_json(folder / STATE_FILE, self.NAME)
        n_bins = int(payload["n_bins"])
        edges = decode_edges(payload["size_bin_edges"])
        if len(edges) != n_bins + 1:
            raise ValueError(
                f"support/{STATE_FILE} declares n_bins={n_bins} but has "
                f"{len(edges)} bin edges."
            )
        self._sorted_self_distances_per_bin = load_per_bin(
            require_file(folder / DISTANCES_FILE, self.NAME), n_bins
        )
        self._size_bin_edges = edges
        self._reference_support = float(payload["reference_support"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._knn is not None
            and self._size_bin_edges is not None
            and self._sorted_self_distances_per_bin is not None
            and self._reference_support is not None
        )

    @property
    def n_bins_(self) -> int:
        """Number of fingerprint-size bins in the fitted calibration."""
        self._check_fitted()
        assert self._sorted_self_distances_per_bin is not None
        return len(self._sorted_self_distances_per_bin)

    @property
    def size_bin_edges_(self) -> np.ndarray:
        self._check_fitted()
        assert self._size_bin_edges is not None
        return self._size_bin_edges

    @property
    def sorted_self_distances_per_bin_(self) -> list[np.ndarray]:
        self._check_fitted()
        assert self._sorted_self_distances_per_bin is not None
        return self._sorted_self_distances_per_bin

    @property
    def reference_support_(self) -> float:
        self._check_fitted()
        assert self._reference_support is not None
        return self._reference_support

    def _get_vector_index(self) -> VectorIndex:
        assert self._shared is not None
        if self._vector_index_cache is None:
            self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache
