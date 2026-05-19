"""Consistency score: how noisy the output-space neighborhood is.

Per-query score is ``exp(−mean(output_space_L1_distances_to_k_neighbors))``,
strictly in ``(0, 1]``. High = neighbors predict similarly to the query
(low noise); low = the query sits in a noisy neighborhood where similar
molecules disagree on the outputs.

Operates in **output space**, on absolute distances — no calibration
against the reference population, no reference baseline. Support is the
sibling that handles chemical proximity (FP-space, CDF-calibrated).
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
from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import (
    _component_metadata,
    _make_pipeline,
    _query_fp_distances,
    _query_output_distances,
    _resolve_shared_and_knn,
    _resolve_vector_index,
)
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex


SUBFOLDER = "consistency"
METADATA_FILE = "metadata.json"


@dataclass
class ConsistencyRunResult:
    """Result returned by :meth:`Consistency.run`."""

    score: pd.Series  # (n_query,) consistency in (0, 1]
    distance_k_mean: pd.Series  # mean output-space L1 to k neighbors
    metadata: dict[str, Any] = field(default_factory=dict)


class Consistency:
    """Output-space neighborhood-noise scorer.

    Holds two pieces of fitted state:

    - ``shared_`` — the shared schema and scaler params.
    - ``knn_`` — the shared :class:`KnnFitState` (``ref_repr`` and
      ``k``); ``ref_repr`` is what gets indexed at run time to compute
      output-space distances against the query's FP neighbors.

    No reference-side calibration: the score is absolute (a function of
    the query's neighborhood alone), so there is no
    ``reference_consistency_`` baseline.
    """

    def __init__(self) -> None:
        self._shared: SharedFitState | None = None
        self._knn: KnnFitState | None = None
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
    ) -> "Consistency":
        """Fit on a reference DataFrame.

        Resolves the shared and kNN states (fitting them if not provided)
        and records fit-time bookkeeping (timestamp + duration). The
        score is absolute, so there is no per-population computation
        beyond what's already in :class:`KnnFitState`.

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

        self._shared = shared
        self._knn = knn
        self._vector_index_cache = vi
        self._fit_duration_seconds = float(time.perf_counter() - t0)
        self._fit_timestamp = datetime.now(tz=timezone.utc).isoformat()
        logger.debug(
            f"Consistency fit | k={knn.k} | n_ref={len(shared.reference_ids):,}"
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
        query_output_distances: np.ndarray | None = None,
    ) -> ConsistencyRunResult:
        """Score query samples.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference plus
            an ``'input'`` SMILES column for the vector index.
        query_repr:
            Optional pre-scaled query array (output of the eosframes
            scaler) ``(n_query, n_features)``. If provided, the eosframes
            transform step is skipped.
        query_fp_indices:
            Optional pre-computed FP-selected neighbor indices
            ``(n_query, k)``.
        query_output_distances:
            Optional pre-computed output-space L1 distances to those
            neighbors ``(n_query, k)``. Used by :class:`ErsiliaQuality`
            to share the kNN work across scores.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._knn is not None

        validate_against_schema(query, self._shared.schema)
        if "input" not in query.columns:
            raise ValueError(
                "Consistency.run requires an 'input' column with SMILES for the vector index."
            )

        if query_repr is None:
            query_repr = _make_pipeline(self._shared).transform(query)
        if query_output_distances is None:
            if query_fp_indices is None:
                _, query_fp_indices = _query_fp_distances(
                    query, self._get_vector_index(), self._knn.k
                )
            query_output_distances = _query_output_distances(
                query_repr, self._knn, query_fp_indices
            )

        distance_k_mean = query_output_distances.mean(axis=1)
        score = np.exp(-distance_k_mean)

        idx = list(query.index)
        return ConsistencyRunResult(
            score=pd.Series(score, index=idx, name="consistency"),
            distance_k_mean=pd.Series(
                distance_k_mean, index=idx, name="distance_k_mean"
            ),
            metadata={
                "n_reference": len(self._shared.reference_ids),
                "k": int(self._knn.k),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, root: str | pathlib.Path) -> pathlib.Path:
        """Persist into ``<root>/shared/``, ``<root>/knn/`` and ``<root>/consistency/``.

        Writes ``consistency/metadata.json`` only (fit timestamp, fit
        duration, k, n_samples, n_features, eosquality_version). There is
        no separate ``state.json`` — Consistency has no calibration data
        of its own; everything it needs at run time lives in the shared
        and kNN subfolders.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._knn is not None
        save_shared(self._shared, root)
        save_knn(self._knn, root)
        folder = pathlib.Path(root) / SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)
        meta = _component_metadata(
            component="consistency",
            k=int(self._knn.k),
            fit_timestamp=self._fit_timestamp,
            fit_duration_seconds=self._fit_duration_seconds,
        )
        with open(folder / METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug(
            f"  consistency/ | fit_duration={meta['fit_duration_seconds']:.3f}s"
        )
        return pathlib.Path(root)

    @classmethod
    def load(cls, root: str | pathlib.Path) -> "Consistency":
        """Reconstruct from ``<root>/shared/`` + ``<root>/knn/`` + ``<root>/consistency/``."""
        shared = load_shared(root)
        knn = load_knn(root)
        folder = pathlib.Path(root) / SUBFOLDER
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
        instance._fit_duration_seconds = fit_duration
        instance._fit_timestamp = fit_timestamp
        return instance

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return self._shared is not None and self._knn is not None

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
    def fit_duration_seconds_(self) -> float | None:
        return self._fit_duration_seconds

    @property
    def fit_timestamp_(self) -> str | None:
        return self._fit_timestamp

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Consistency must be fitted (or loaded) before use.")

    def _get_vector_index(self) -> VectorIndex:
        assert self._shared is not None
        if self._vector_index_cache is None:
            self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache
