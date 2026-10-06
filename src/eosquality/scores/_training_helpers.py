"""Private helpers shared by the training-modality scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eosquality.scores._helpers import (
    _query_fp_distances_valid,
    _row_nanquantile,
    _standardize,
)
from eosquality.vectorindex import VectorIndex

# Quantile across output columns for the whole-model value (the column
# analogue of the Q66 aggregate typicality and extremity use over features).
SUMMARY_QUANTILE = 0.66


def _columns_summary(values: np.ndarray) -> np.ndarray:
    """NaN-ignoring ``SUMMARY_QUANTILE`` across columns (NaN for all-NaN rows)."""
    return _row_nanquantile(values, SUMMARY_QUANTILE)


class TrainingQuery:
    """Query features computed once per run and shared by the training scores.

    Standardising SMILES, MACCS keys, Morgan bits and each column's
    nearest-neighbour search are the costly steps of scoring; training
    distance and every error model of training difficulty need the same
    ones, so they are computed on first use and cached here.

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
        self._maccs: np.ndarray | None = None
        self._morgan: np.ndarray | None = None
        self._nearest: dict[str, tuple[int, tuple]] = {}

    @classmethod
    def from_frame(cls, query: pd.DataFrame) -> TrainingQuery:
        """Standardise the ``input`` column; unparsable rows are left out.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column.

        Returns
        -------
        TrainingQuery
        """
        if "input" not in query.columns:
            raise ValueError("The training scores need an 'input' SMILES column.")
        std = [_standardize(s) for s in query["input"]]
        rows = np.flatnonzero([s is not None for s in std])
        return cls([std[i] for i in rows], rows, len(query))

    @property
    def maccs(self) -> np.ndarray:
        """``(n, 166)`` MACCS keys.

        Returns
        -------
        numpy.ndarray
        """
        if self._maccs is None:
            from eosquality.library.maccs import compute_maccs

            self._maccs = compute_maccs(self.smiles, show_progress=False)
        return self._maccs

    @property
    def morgan(self) -> np.ndarray:
        """``(n, 2048)`` Morgan bits.

        Returns
        -------
        numpy.ndarray
        """
        if self._morgan is None:
            from eosquality.training.folds import morgan_bits

            self._morgan = morgan_bits(self.smiles)
        return self._morgan

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
    """Top-k training neighbours of standardised SMILES (see ``TrainingQuery``)."""
    if not query_smiles:
        return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64), np.zeros(0, bool)
    dist, nn = _query_fp_distances_valid(query_smiles, vi, k, exclude_self_match=True)
    members = set(vi.smiles)
    hit = np.array([smi in members for smi in query_smiles])
    return dist, nn, hit
