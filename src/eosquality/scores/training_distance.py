"""Training distance: how far is the query from each output column's training set?

Training mode, X only. For every output column with a training set, the raw
distance is ``1 − mean Tanimoto similarity`` (Morgan, radius 2, 2048 bits)
to the query's **k = 5 nearest training molecules** (Sheridan et al. 2004,
"mean similarity to the 5 nearest neighbours"). A query that is itself a
training molecule (same standardised SMILES: largest fragment, canonical
isomeric) drops its own entry, so it gets its leave-one-out value and is
flagged ``in_training``.

The calibrated distance is the mid-rank CDF of the raw value against the
column's own leave-one-out raw values (each training molecule vs its k
nearest *other* training molecules): ~0.5 for a query as close to the
training set as a typical training molecule, near 1 when farther than almost
all of them. Higher is farther. There is no in/out cutoff.

The per-molecule summaries ``training_distance`` and
``training_distance_raw`` are the 66th percentile across columns of the
calibrated and the raw values: at least two-thirds of the columns are this
close or closer. Per-column values and the nearest training molecules
(keys, similarities, labels) go to ``TrainingDistanceRunResult.details``.
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

from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import (
    _cdf_score,
    _query_fp_distances,
    _sorted_finite,
    _standardize,
)
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState
from eosquality.utils.logging import logger

SUBFOLDER = "training_distance"
STATE_FILE = "state.json"
LOO_FILE = "loo_mean_distances.npz"
# Nearest training neighbours averaged per (molecule, column), and reported
# in the details table. Capped by the column's precomputed self-kNN.
K_NEIGHBORS = 5
# Quantile across columns for the per-molecule summary (the distance analogue
# of the Q66 aggregate typicality and extremity use over features).
SUMMARY_QUANTILE = 0.66

DETAIL_COLUMNS = [
    "key",
    "column",
    "distance",
    "distance_raw",
    "nn1_distance",
    "k",
    "n_train",
    "in_training",
    "nn_keys",
    "nn_similarities",
    "nn_y",
]


@dataclass
class TrainingDistanceRunResult:
    """Result returned by :meth:`TrainingDistance.run`."""

    score: pd.Series  # (n_query,) calibrated summary across columns, in (0, 1]
    score_raw: pd.Series  # (n_query,) raw mean-k distance summary, in [0, 1]
    n_columns: pd.Series  # (n_query,) columns contributing to the summary
    in_training_any: pd.Series  # (n_query,) query is a training molecule of a column
    details: pd.DataFrame  # one row per (query, column)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingDistance(ScoreComponent):
    """Per-column mean Morgan distance to the k nearest training molecules."""

    NAME = SUBFOLDER
    USES_SHARED = False  # X only: needs the training sets, not the reference
    USES_TRAINING = True

    def __init__(self) -> None:
        super().__init__()
        self._columns: list[str] | None = None
        self._k: dict[str, int] | None = None
        self._loo: dict[str, np.ndarray] | None = None  # column → sorted LOO raw

    def fit(
        self, *, training: TrainingFitState, shared: SharedFitState | None = None
    ) -> TrainingDistance:
        """Build each column's leave-one-out calibration table.

        Parameters
        ----------
        training : TrainingFitState
            Training sets and their per-column Morgan indices.
        shared : SharedFitState, optional
            Unused; accepted for a uniform component interface.

        Returns
        -------
        TrainingDistance
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        self._training = training
        self._columns = training.column_names
        self._k, self._loo = {}, {}
        for name in self._columns:
            k = min(K_NEIGHBORS, training.columns[name].n - 2)
            loo = training.indices[name].self_knn_distances(k).mean(axis=1)
            self._k[name] = k
            self._loo[name] = _sorted_finite(loo, self.NAME)
        self._finish_fit(t0)
        logger.debug(f"TrainingDistance fit | {len(self._columns)} columns")
        return self

    def run(self, query: pd.DataFrame) -> TrainingDistanceRunResult:
        """Distance of each query to every column's training set.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column; ``key`` (if present) labels the
            rows of the details table.

        Returns
        -------
        TrainingDistanceRunResult
            Calibrated and raw summaries, per-column details and metadata.
        """
        self._check_fitted()
        assert self._training is not None and self._k is not None
        assert self._loo is not None
        if "input" not in query.columns:
            raise ValueError("TrainingDistance.run requires an 'input' SMILES column.")
        idx = list(query.index)
        keys = (
            query["key"].astype(str).tolist()
            if "key" in query.columns
            else [str(i) for i in idx]
        )
        std = [_standardize(s) for s in query["input"]]
        rows = np.flatnonzero([s is not None for s in std])
        names = self._training.column_names
        raw = np.full((len(query), len(names)), np.nan)
        calibrated = np.full_like(raw, np.nan)
        in_train = np.zeros(raw.shape, dtype=bool)
        details = []
        for j, name in enumerate(names):
            dist, nn_idx, hit = _nearest_training(
                self._training.indices[name], [std[i] for i in rows], self._k[name]
            )
            raw[rows, j] = dist.mean(axis=1)
            calibrated[rows, j] = _cdf_score(
                raw[rows, j], self._loo[name], higher_is_higher=True
            )
            in_train[rows, j] = hit
            details.append(
                _details_rows(
                    [keys[i] for i in rows],
                    self._training.columns[name],
                    calibrated[rows, j],
                    dist,
                    nn_idx,
                    hit,
                )
            )
        return TrainingDistanceRunResult(
            score=pd.Series(_summary(calibrated), index=idx, name="training_distance"),
            score_raw=pd.Series(_summary(raw), index=idx, name="training_distance_raw"),
            n_columns=pd.Series(
                np.isfinite(raw).sum(axis=1), index=idx, name="training_n_columns"
            ),
            in_training_any=pd.Series(
                in_train.any(axis=1), index=idx, name="in_training_any"
            ),
            details=pd.concat(details, ignore_index=True)
            if details
            else pd.DataFrame(columns=DETAIL_COLUMNS),
            metadata={
                "n_columns": len(names),
                "columns": names,
                "k": K_NEIGHBORS,
                "summary_quantile": SUMMARY_QUANTILE,
            },
        )

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._columns is not None and self._k is not None
        assert self._loo is not None
        np.savez(
            folder / LOO_FILE,
            **{f"c{j:03d}": self._loo[n] for j, n in enumerate(self._columns)},
        )
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"columns": self._columns, "k": self._k}, f, indent=2)

    def _load_own(self, folder: pathlib.Path) -> None:
        assert self._training is not None
        state = read_json(folder / STATE_FILE, self.NAME)
        names = state["columns"]
        if names != self._training.column_names:
            raise ValueError(
                "training_distance/state.json columns do not match "
                "training_sets/metadata.json."
            )
        with np.load(require_file(folder / LOO_FILE, self.NAME)) as npz:
            self._loo = {n: npz[f"c{j:03d}"] for j, n in enumerate(names)}
        self._k = {n: int(state["k"][n]) for n in names}
        self._columns = names

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return self._training is not None and self._loo is not None

    @property
    def training_(self) -> TrainingFitState:
        """The training sets the distances are computed against.

        Returns
        -------
        TrainingFitState
        """
        self._check_fitted()
        assert self._training is not None
        return self._training


def _summary(values: np.ndarray) -> np.ndarray:
    """NaN-ignoring ``SUMMARY_QUANTILE`` across columns (NaN for all-NaN rows)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
        return np.nanquantile(values, SUMMARY_QUANTILE, axis=1)


def _details_rows(keys, column, calibrated, dist, nn_idx, hit) -> pd.DataFrame:
    """One details row per query for one training column."""
    ids = column.ids
    return pd.DataFrame(
        {
            "key": keys,
            "column": column.name,
            "distance": calibrated,
            "distance_raw": dist.mean(axis=1) if len(dist) else [],
            "nn1_distance": dist[:, 0] if len(dist) else [],
            "k": dist.shape[1],
            "n_train": column.n,
            "in_training": hit,
            "nn_keys": ["|".join(ids[t] for t in row) for row in nn_idx],
            "nn_similarities": ["|".join(f"{1 - v:.3f}" for v in row) for row in dist],
            "nn_y": [
                "|".join(_fmt(column.y[t]) for t in row) if column.y is not None else ""
                for row in nn_idx
            ],
        },
        columns=DETAIL_COLUMNS,
    )


def _nearest_training(vi, query_smiles: list[str], k: int):
    """Top-k training neighbours per query (closest first), self excluded.

    A query that is a training molecule drops its own entry, so it is scored
    like the leave-one-out calibration table. Returns
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


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"
