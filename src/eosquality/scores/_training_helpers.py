"""Private helpers shared by the training-modality scores."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from eosquality.scores._helpers import _query_fp_distances
from eosquality.vectorindex import VectorIndex

# Quantile across output columns for the whole-model value (the column
# analogue of the Q66 aggregate typicality and extremity use over features).
SUMMARY_QUANTILE = 0.66


def _columns_summary(values: np.ndarray) -> np.ndarray:
    """NaN-ignoring ``SUMMARY_QUANTILE`` across columns (NaN for all-NaN rows)."""
    if values.shape[1] == 0:
        return np.full(values.shape[0], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        return np.nanquantile(values, SUMMARY_QUANTILE, axis=1)


def _nearest_training(vi: VectorIndex, query_smiles: list[str], k: int):
    """Top-k training neighbours per query (closest first), self excluded.

    A query that is a training molecule drops its own entry, so it is scored
    like the leave-one-out tables built from the index's self-kNN. Returns
    ``(distances (n, k), indices (n, k), in_training (n,))``, where
    ``in_training`` marks queries whose standardised SMILES is a training
    molecule of this column.
    """
    if not query_smiles:
        return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64), np.zeros(0, bool)
    dist, nn = _query_fp_distances(pd.DataFrame({"input": query_smiles}), vi, k)
    members = set(vi.smiles)
    hit = np.array([smi in members for smi in query_smiles])
    return dist, nn, hit
