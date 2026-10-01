"""Support score: where the query sits in the reference's FP self-distance CDF.

Closer than every reference point → ~1.0; at the reference median → ~0.5;
farther than every reference point → eps.

Operates in **fingerprint space**: both the calibration CDF (built from
the reference's own k FP-nearest neighbors at fit time) and the per-query
score (mean Tanimoto distance to the query's k FP neighbors) use the same
Tanimoto metric. Consistency is the sibling that lives in output space.
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
from eosquality.scores._helpers import (
    _cdf_score,
    _query_fp_distances,
    _resolve_shared_and_knn,
    _resolve_vector_index,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

SUBFOLDER = "support"
STATE_FILE = "state.json"
DISTANCES_FILE = "reference_self_distances.npy"


@dataclass
class SupportRunResult:
    """Result returned by :meth:`Support.run`."""

    score: pd.Series  # (n_query,) calibrated support in [0, 1]
    score_raw: pd.Series  # (n_query,) raw mean FP Tanimoto distance (= distance_k_mean)
    distance_k_mean: pd.Series  # mean FP (Tanimoto) distance to k neighbors
    distance_k_max: pd.Series  # max FP (Tanimoto) distance to k neighbors
    nearest_reference_ids: list[list[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class Support(ScoreComponent):
    """CDF-based support scorer (FP-space Tanimoto distances).

    Holds three pieces of fitted state:

    - ``sorted_self_distances_`` — ``(n_ref,)`` ascending array of mean
      FP (Tanimoto) k-distances. The CDF lookup table.
    - ``reference_support_`` — mean reference-as-query support under that
      CDF. A calibration anchor for downstream readers.
    - ``knn_`` — the shared :class:`KnnFitState` (just ``k`` after
      save/load; the fit-only ``mean_fp_distances`` and
      ``reference_knn_indices`` are dropped).

    Depends on both :class:`SharedFitState` and :class:`KnnFitState`; loads
    the underlying :class:`VectorIndex` lazily on first call to
    :meth:`run`. Support never reads ``shared.ref_repr`` — it operates
    on Tanimoto distances only.
    """

    NAME = SUBFOLDER
    USES_KNN = True

    def __init__(self) -> None:
        super().__init__()
        self._sorted_self_distances: np.ndarray | None = None
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

        Reads the reference's FP self-kNN Tanimoto distances (already
        identity-stripped at vector-index build time), sorts the per-row
        mean FP distances to form the CDF lookup table, and records
        ``reference_support_`` (mean reference-as-query support) as a
        calibration baseline.

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
        sorted_self_distances = np.sort(knn.mean_fp_distances).astype(np.float64)

        self._shared = shared
        self._knn = knn
        self._sorted_self_distances = sorted_self_distances
        # Same formula as run() applied to the reference itself; ≈ 0.5 for a
        # healthy reference.
        self._reference_support = float(
            np.mean(
                _support_from_distances(knn.mean_fp_distances, sorted_self_distances)
            )
        )
        self._vector_index_cache = vi
        self._finish_fit(t0)
        logger.debug(
            f"Support fit | k={knn.k} | n_ref={len(knn.mean_fp_distances):,}"
            f" | reference_support={self._reference_support:.4f}"
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
        assert self._sorted_self_distances is not None

        if "input" not in query.columns:
            raise ValueError(
                "Support.run requires an 'input' column with SMILES for the vector index."
            )

        if query_fp_indices is None or query_fp_distances is None:
            query_fp_distances, query_fp_indices = _query_fp_distances(
                query, self._get_vector_index(), self._knn.k
            )

        n_ref = len(self._shared.reference_ids)
        distance_k_mean = query_fp_distances.mean(axis=1)
        support_score = _support_from_distances(
            distance_k_mean, self._sorted_self_distances
        )

        idx = list(query.index)
        reference_ids = self._shared.reference_ids
        return SupportRunResult(
            score=pd.Series(support_score, index=idx, name="support"),
            score_raw=pd.Series(distance_k_mean, index=idx, name="support_raw"),
            distance_k_mean=pd.Series(
                distance_k_mean, index=idx, name="distance_k_mean"
            ),
            distance_k_max=pd.Series(
                query_fp_distances.max(axis=1), index=idx, name="distance_k_max"
            ),
            nearest_reference_ids=[
                [reference_ids[j] for j in row] for row in query_fp_indices
            ],
            metadata={
                "reference_support": self._reference_support,
                "n_reference": n_ref,
                "k": int(self._knn.k),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline) and the sorted CDF array."""
        assert self._sorted_self_distances is not None
        np.save(folder / DISTANCES_FILE, self._sorted_self_distances)
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"reference_support": self._reference_support}, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        self._sorted_self_distances = np.load(
            require_file(folder / DISTANCES_FILE, self.NAME)
        )
        payload = read_json(folder / STATE_FILE, self.NAME)
        self._reference_support = float(payload["reference_support"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._knn is not None
            and self._sorted_self_distances is not None
            and self._reference_support is not None
        )

    @property
    def sorted_self_distances_(self) -> np.ndarray:
        self._check_fitted()
        assert self._sorted_self_distances is not None
        return self._sorted_self_distances

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


# ---------------------------------------------------------------------------
# Support-specific helpers
# ---------------------------------------------------------------------------


def _support_from_distances(
    distance_k_mean: np.ndarray, sorted_self_distances: np.ndarray
) -> np.ndarray:
    """Map per-row mean k-distances to support scores via the reference CDF.

    Distance-direction wrapper around :func:`_cdf_score`: a smaller FP
    distance (closer to the reference) gives higher support.
    """
    return _cdf_score(distance_k_mean, sorted_self_distances, higher_is_higher=False)
