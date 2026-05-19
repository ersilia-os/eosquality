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
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from eosquality.knn.load import load_knn
from eosquality.knn.save import save_knn
from eosquality.knn.state import KnnFitState
from eosquality.scores._helpers import (
    _component_metadata,
    _query_fp_distances,
    _resolve_shared_and_knn,
    _resolve_vector_index,
)
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex


SUBFOLDER = "support"
STATE_FILE = "state.json"
DISTANCES_FILE = "reference_self_distances.npy"
METADATA_FILE = "metadata.json"


@dataclass
class SupportRunResult:
    """Result returned by :meth:`Support.run`."""

    score: pd.Series  # (n_query,) support
    distance_k_mean: pd.Series  # mean FP (Tanimoto) distance to k neighbors
    distance_k_max: pd.Series  # max FP (Tanimoto) distance to k neighbors
    nearest_reference_ids: list[list[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class Support:
    """CDF-based support scorer (FP-space Tanimoto distances).

    Holds three pieces of fitted state:

    - ``sorted_self_distances_`` — ``(n_ref,)`` ascending array of mean
      FP (Tanimoto) k-distances. The CDF lookup table.
    - ``reference_support_`` — mean reference-as-query support under that
      CDF. A calibration anchor for downstream readers.
    - ``knn_`` — the shared :class:`KnnFitState` (``ref_repr`` and
      ``k``).

    Depends on both :class:`SharedFitState` and :class:`KnnFitState`; loads
    the underlying :class:`VectorIndex` lazily on first call to
    :meth:`run`.
    """

    def __init__(self) -> None:
        self._shared: SharedFitState | None = None
        self._knn: KnnFitState | None = None
        self._sorted_self_distances: np.ndarray | None = None
        self._reference_support: float | None = None
        self._vector_index_cache: VectorIndex | None = None
        self._fit_duration_seconds: float | None = None
        self._fit_timestamp: str | None = None

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
    ) -> "Support":
        """Fit on a reference DataFrame.

        Reads the reference's FP self-kNN Tanimoto distances (already
        identity-stripped at vector-index build time), sorts the per-row
        mean FP distances to form the CDF lookup table, and records
        ``reference_support_`` (mean reference-as-query support) as a
        calibration baseline.

        Either pass pre-fit ``shared=`` / ``knn=`` (when composed by
        :class:`ErsiliaQuality`), or pass ``eos_id`` + ``version`` +
        ``vector_index`` so Support can fit both states itself.

        Records the wall-clock duration and a UTC timestamp; both are
        persisted by :meth:`save` to ``support/metadata.json``.
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
        # reference_support_ is the mean support a reference row receives
        # against its own CDF — same formula as run() applied to the
        # reference itself. By construction this is ≈ 0.5 for a healthy
        # reference; large drifts signal heavy ties in self-distances.
        reference_support = float(
            np.mean(
                _support_from_distances(
                    knn.mean_fp_distances,
                    sorted_self_distances,
                    len(knn.mean_fp_distances),
                )
            )
        )

        self._shared = shared
        self._knn = knn
        self._sorted_self_distances = sorted_self_distances
        self._reference_support = reference_support
        self._vector_index_cache = vi
        self._fit_duration_seconds = float(time.perf_counter() - t0)
        self._fit_timestamp = datetime.now(tz=timezone.utc).isoformat()
        logger.debug(
            f"Support fit | k={knn.k} | n_ref={len(knn.mean_fp_distances):,}"
            f" | reference_support={reference_support:.4f}"
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
        query_fp_indices:
            Optional pre-computed FP-selected neighbor indices
            ``(n_query, k)``.
        query_fp_distances:
            Optional pre-computed Tanimoto distances to those neighbors
            ``(n_query, k)``. ``query_fp_indices`` and
            ``query_fp_distances`` must be passed together or not at all
            — used by :class:`ErsiliaQuality` to share the FP query
            across scores.
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
        distance_k_max = query_fp_distances.max(axis=1)
        support_score = _support_from_distances(
            distance_k_mean, self._sorted_self_distances, n_ref
        )

        idx = list(query.index)
        nearest_reference_ids = [
            [self._shared.reference_ids[j] for j in query_fp_indices[i]]
            for i in range(len(query))
        ]
        return SupportRunResult(
            score=pd.Series(support_score, index=idx, name="support"),
            distance_k_mean=pd.Series(
                distance_k_mean, index=idx, name="distance_k_mean"
            ),
            distance_k_max=pd.Series(distance_k_max, index=idx, name="distance_k_max"),
            nearest_reference_ids=nearest_reference_ids,
            metadata={
                "reference_support": self._reference_support,
                "n_reference": n_ref,
                "k": int(self._knn.k),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, root: str | pathlib.Path) -> pathlib.Path:
        """Persist into ``<root>/shared/``, ``<root>/knn/`` and ``<root>/support/``.

        Writes three files under ``support/``:

        - ``state.json`` — the ``reference_support`` baseline.
        - ``reference_self_distances.npy`` — the sorted CDF array.
        - ``metadata.json`` — fit timestamp, fit duration, k, n_samples,
          n_features, eosquality_version.

        Also writes the shared and kNN subfolders via :func:`save_shared`
        and :func:`save_knn` so the artifact is self-contained.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._knn is not None
        assert self._sorted_self_distances is not None
        save_shared(self._shared, root)
        save_knn(self._knn, root)
        folder = pathlib.Path(root) / SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / DISTANCES_FILE, self._sorted_self_distances)
        payload = {"reference_support": self._reference_support}
        with open(folder / STATE_FILE, "w") as f:
            json.dump(payload, f)
        meta = _component_metadata(
            component="support",
            k=int(self._knn.k),
            fit_timestamp=self._fit_timestamp,
            fit_duration_seconds=self._fit_duration_seconds,
        )
        with open(folder / METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug(
            f"  support/ | reference_support={self._reference_support:.4f}"
            f" | fit_duration={meta['fit_duration_seconds']:.3f}s"
        )
        return pathlib.Path(root)

    @classmethod
    def load(cls, root: str | pathlib.Path) -> "Support":
        """Reconstruct from ``<root>/shared/`` + ``<root>/knn/`` + ``<root>/support/``."""
        shared = load_shared(root)
        knn = load_knn(root)
        folder = pathlib.Path(root) / SUBFOLDER
        sorted_self_distances = np.load(folder / DISTANCES_FILE)
        with open(folder / STATE_FILE) as f:
            payload = json.load(f)
        meta_path = folder / METADATA_FILE
        fit_duration = None
        fit_timestamp = None
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            fit_duration = float(meta.get("fit_duration_seconds", 0.0))
            fit_timestamp = meta.get("fit_timestamp")
        instance = cls()
        instance._shared = shared
        instance._knn = knn
        instance._sorted_self_distances = sorted_self_distances
        instance._reference_support = float(payload["reference_support"])
        instance._fit_duration_seconds = fit_duration
        instance._fit_timestamp = fit_timestamp
        return instance

    # ------------------------------------------------------------------
    # Properties / helpers
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
    def shared_(self) -> SharedFitState:
        self._check_fitted()
        assert self._shared is not None
        return self._shared

    @property
    def knn_(self) -> KnnFitState:
        self._check_fitted()
        assert self._knn is not None
        return self._knn

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

    @property
    def fit_duration_seconds_(self) -> float | None:
        return self._fit_duration_seconds

    @property
    def fit_timestamp_(self) -> str | None:
        return self._fit_timestamp

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Support must be fitted (or loaded) before use.")

    def _get_vector_index(self) -> VectorIndex:
        assert self._shared is not None
        if self._vector_index_cache is None:
            self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache


# ---------------------------------------------------------------------------
# Support-specific helpers
# ---------------------------------------------------------------------------


def _support_from_distances(
    distance_k_mean: np.ndarray,
    sorted_self_distances: np.ndarray,
    n_reference: int,
) -> np.ndarray:
    """Map per-row mean k-distances to support scores via the CDF.

    ``support = clip(1 − searchsorted(sorted_self_distances, d) / n_reference,
    eps, 1.0)`` with ``eps = 1 / (2·n_reference)``. Shared by
    :meth:`Support.fit` (for the ``reference_support_`` baseline) and
    :meth:`Support.run` so the two paths cannot drift.
    """
    eps = 1.0 / (2.0 * max(n_reference, 1))
    cdf = (
        np.searchsorted(sorted_self_distances, distance_k_mean, side="right")
        / n_reference
    )
    return np.clip(1.0 - cdf, eps, 1.0)
