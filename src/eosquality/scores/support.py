"""Support score: does the reference library contain a close analogue of the query?

The raw value is the Tanimoto similarity (Morgan, radius 2, 2048 bits) of
the query's **nearest library analogue** — the most similar library molecule
other than the query itself (``support_raw``, in [0, 1], higher = closer).
It is calibrated through the library's own nearest-analogue similarities
(each library molecule vs its closest *other* molecule), so library
molecules score ~Uniform(0, 1): a nearer analogue than any library molecule
has → ~1.0, the library median → ~0.5, farther than all → eps.

``support_log = −log10(support)`` re-expresses the same tail probability on
a log scale, so queries far outside the library (where ``support`` is
squeezed into 0–0.01) remain distinguishable.

The raw similarity also reads directly in chemists' terms: below ~0.4 the
library holds no related chemistry, ~0.6 is a close analogue, ≥ 0.8 a
near-identical one. Tanimoto similarity is lower for small molecules (few
set bits), so small fragments look somewhat more novel; this is not
corrected for.

Operates in fingerprint space only. Neighbourhood quality in output space
is Consistency's job (it uses the k nearest neighbours, not just one).
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
SIMILARITIES_FILE = "reference_nearest_similarities.npy"


@dataclass
class SupportRunResult:
    """Result returned by :meth:`Support.run`."""

    score: pd.Series  # (n_query,) calibrated support in (0, 1]
    score_raw: (
        pd.Series
    )  # (n_query,) Tanimoto similarity of the nearest library analogue
    score_log: pd.Series  # (n_query,) −log10(support), ≥ 0
    distance_k_mean: pd.Series  # mean FP (Tanimoto) distance to the k neighbors
    nearest_reference_ids: list[list[Any]]  # k nearest, closest first
    metadata: dict[str, Any] = field(default_factory=dict)


class Support(ScoreComponent):
    """Nearest-analogue support scorer (FP-space Tanimoto similarity).

    Fitted state on top of the shared and kNN states:

    - ``sorted_self_similarities_`` — ``(n_ref,)`` ascending nearest-analogue
      similarities of the library; the CDF lookup table.
    - ``reference_support_`` — mean reference-as-query support (≈ 0.5).

    Loads the underlying :class:`VectorIndex` lazily on first :meth:`run`.
    """

    NAME = SUBFOLDER
    USES_KNN = True

    def __init__(self) -> None:
        super().__init__()
        self._sorted_self_similarities: np.ndarray | None = None
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
        """Build the calibration table of nearest-analogue similarities.

        Pass pre-fit ``shared`` / ``knn`` (as :class:`ErsiliaQuality` does), or
        ``eos_id`` + ``version`` so both states are fitted here.

        Parameters
        ----------
        reference : pandas.DataFrame
            Predictions on the reference library.
        vector_index : str, pathlib.Path or VectorIndex
            The reference library's vector index.
        k : int, optional
            Fingerprint neighbours reported per query.
        eos_id, version : str, optional
            Model id and version, needed only to fit ``shared`` here.
        shared : SharedFitState, optional
            Pre-fit shared state.
        knn : KnnFitState, optional
            Pre-fit kNN state.

        Returns
        -------
        Support
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
        similarities = 1.0 - vi.self_knn_distances(1)[:, 0].astype(np.float64)
        sorted_self = np.sort(similarities)

        self._shared = shared
        self._knn = knn
        self._sorted_self_similarities = sorted_self
        self._reference_support = float(
            np.mean(_cdf_score(similarities, sorted_self, higher_is_higher=True))
        )
        self._vector_index_cache = vi
        self._finish_fit(t0)
        logger.debug(
            f"Support fit | n_ref={len(similarities):,} | "
            f"reference_support={self._reference_support:.4f}"
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
        """Score query molecules.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column; other columns are ignored.
        query_fp_indices, query_fp_distances : numpy.ndarray, optional
            Pre-computed ``(n_query, k)`` FP neighbours and Tanimoto distances
            (pass both or neither); recomputed from the vector index if omitted.

        Returns
        -------
        SupportRunResult
            Calibrated score, nearest-analogue similarity, ``support_log``,
            neighbour ids and metadata.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._knn is not None
        assert self._sorted_self_similarities is not None

        if "input" not in query.columns:
            raise ValueError(
                "Support.run requires an 'input' column with SMILES for the vector index."
            )
        if query_fp_indices is None or query_fp_distances is None:
            query_fp_distances, query_fp_indices = _query_fp_distances(
                query, self._get_vector_index(), self._knn.k
            )

        nearest_similarity = 1.0 - query_fp_distances.min(axis=1)
        support_score = _cdf_score(
            nearest_similarity, self._sorted_self_similarities, higher_is_higher=True
        )

        idx = list(query.index)
        reference_ids = self._shared.reference_ids
        return SupportRunResult(
            score=pd.Series(support_score, index=idx, name="support"),
            score_raw=pd.Series(nearest_similarity, index=idx, name="support_raw"),
            score_log=pd.Series(
                -np.log10(support_score), index=idx, name="support_log"
            ),
            distance_k_mean=pd.Series(
                query_fp_distances.mean(axis=1), index=idx, name="distance_k_mean"
            ),
            nearest_reference_ids=[
                [reference_ids[j] for j in row] for row in query_fp_indices
            ],
            metadata={
                "reference_support": self._reference_support,
                "n_reference": len(reference_ids),
                "k": int(self._knn.k),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline) and the sorted CDF array."""
        assert self._sorted_self_similarities is not None
        np.save(folder / SIMILARITIES_FILE, self._sorted_self_similarities)
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"reference_support": self._reference_support}, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        self._sorted_self_similarities = np.load(
            require_file(folder / SIMILARITIES_FILE, self.NAME)
        )
        payload = read_json(folder / STATE_FILE, self.NAME)
        self._reference_support = float(payload["reference_support"])

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
            and self._sorted_self_similarities is not None
            and self._reference_support is not None
        )

    @property
    def sorted_self_similarities_(self) -> np.ndarray:
        """Sorted nearest-analogue similarities of the library (the calibration CDF).

        Returns
        -------
        numpy.ndarray
        """
        self._check_fitted()
        assert self._sorted_self_similarities is not None
        return self._sorted_self_similarities

    @property
    def reference_support_(self) -> float:
        """Mean calibrated support of the reference (about 0.5).

        Returns
        -------
        float
        """
        self._check_fitted()
        assert self._reference_support is not None
        return self._reference_support

    def _get_vector_index(self) -> VectorIndex:
        assert self._shared is not None
        if self._vector_index_cache is None:
            self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache
