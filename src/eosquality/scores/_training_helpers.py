"""Private helpers shared by the training-modality scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eosquality.scores._helpers import _row_nanquantile, _standardize_all
from eosquality.scores._match_keys import _layers
from eosquality.vectorindex import VectorIndex

# Quantile across output columns for the whole-model value (the column
# analogue of the Q66 aggregate typicality and extremity use over features).
SUMMARY_QUANTILE = 0.66
# Nearest training neighbours averaged per (molecule, column), for the Morgan
# and the physchem distance alike. Capped by the column's size.
K_NEIGHBORS = 5
# A Tanimoto distance below this is a perfect fingerprint match. FPSim2 returns
# exactly 0.0; the epsilon guards against float wobble.
_SELF_MATCH_DISTANCE = 1e-6
# A progress bar is drawn for query descriptor passes of at least this many molecules.
PROGRESS_MIN_MOLECULES = 500


def _columns_summary(values: np.ndarray) -> np.ndarray:
    """NaN-ignoring ``SUMMARY_QUANTILE`` across columns (NaN for all-NaN rows)."""
    return _row_nanquantile(values, SUMMARY_QUANTILE)


class QueryFeatures:
    """Query features computed once per run and shared by the scores that need them.

    Standardising SMILES, the InChIKey layers, physchem descriptors and each
    column's nearest-neighbour search are the costly steps of scoring; the match
    and training scores need the same ones, so they are computed on first use
    and cached here.

    Parameters
    ----------
    smiles : list of str
        Standardised, valid query SMILES.
    rows : numpy.ndarray, optional
        Their positions in the query frame (default: ``0 … n − 1``).
    n_rows : int, optional
        Number of rows of the query frame (default: ``len(smiles)``).
    """

    def __init__(
        self,
        smiles: list[str],
        rows: np.ndarray | None = None,
        n_rows: int | None = None,
    ) -> None:
        self.smiles = list(smiles)
        self.rows = np.arange(len(self.smiles)) if rows is None else rows
        self.n_rows = len(self.smiles) if n_rows is None else n_rows
        self._physchem: np.ndarray | None = None
        self._layers: tuple[np.ndarray, np.ndarray] | None = None
        self._nearest: dict[str, tuple[int, tuple]] = {}

    @classmethod
    def from_frame(cls, query: pd.DataFrame) -> QueryFeatures:
        """Standardise the ``input`` column; unparsable rows are left out.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column.

        Returns
        -------
        QueryFeatures
        """
        if "input" not in query.columns:
            raise ValueError("The training scores need an 'input' SMILES column.")
        std = _standardize_all(list(query["input"]))
        rows = np.flatnonzero([s is not None for s in std])
        return cls([std[i] for i in rows], rows, len(query))

    @property
    def layers(self) -> tuple[np.ndarray, np.ndarray]:
        """InChIKey connectivity layers of the molecules and of their scaffolds.

        Returns
        -------
        tuple of numpy.ndarray
            ``(molecules, scaffolds)``; the match keys of ``_match_keys``.
        """
        if self._layers is None:
            self._layers = _layers(self.smiles, "query InChIKey layers")
        return self._layers

    @property
    def physchem(self) -> np.ndarray:
        """``(n, 217)`` raw RDKit physchem descriptors; non-finite cells allowed.

        Returns
        -------
        numpy.ndarray
        """
        if self._physchem is None:
            from eosquality.library.physchem_cache import describe

            self._physchem = describe(
                self.smiles,
                show_progress=len(self.smiles) >= PROGRESS_MIN_MOLECULES,
                label="query physchem descriptors",
            )
        return self._physchem

    def nearest(self, vi: VectorIndex, k: int):
        """Top-k training neighbours in ``vi`` (closest first), self excluded.

        A query that is a training molecule drops its own entry, so it is
        scored like the leave-one-out tables built from the index's
        self-kNN. A search with a larger ``k`` serves smaller ones.

        Parameters
        ----------
        vi : VectorIndex
            One column's training index.
        k : int
            Neighbours per query.

        Returns
        -------
        tuple
            ``(distances (n, k), indices (n, k), in_training (n,))``;
            ``in_training`` marks queries that are a training molecule of
            this column.
        """
        key = str(vi.index_dir)  # one index per training column
        cached = self._nearest.get(key)
        if cached is None or cached[0] < k:
            cached = (k, _nearest_training(vi, self.smiles, k))
            self._nearest[key] = cached
        dist, nn, hit = cached[1]
        return dist[:, :k], nn[:, :k], hit


def _nearest_training(vi: VectorIndex, query_smiles: list[str], k: int):
    """Top-k training neighbours of standardised SMILES (see ``QueryFeatures``).

    The index is searched for ``k + 1`` and one entry per query is dropped: the
    query's own entry when it is a training molecule (the same standardised
    SMILES; a different molecule with an identical fingerprint stays a
    neighbour), else the furthest.
    """
    n = len(query_smiles)
    if not n:
        return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64), np.zeros(0, bool)
    dist, nn = vi.query(query_smiles, k=k + 1)
    dist = dist.astype(np.float64)
    library = vi.smiles
    drop = np.full(n, k, dtype=np.int64)
    for i, j in zip(*np.nonzero(dist < _SELF_MATCH_DISTANCE), strict=True):
        if drop[i] == k and library[nn[i, j]] == query_smiles[i]:
            drop[i] = j
    keep = np.arange(k + 1)[None, :] != drop[:, None]
    members = set(library)
    hit = np.array([smi in members for smi in query_smiles])
    return dist[keep].reshape(n, k), nn[keep].reshape(n, k), hit
