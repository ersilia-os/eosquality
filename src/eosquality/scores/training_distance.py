"""Training distance: how far is the query from each output column's training set?

Training mode, X only, **not calibrated**. For every output column with a
training set, the distance is ``1 − Tanimoto similarity`` (Morgan, radius 2,
2048 bits) between the query and its **nearest training molecule**: 0 when
the query is itself a training molecule (same standardised SMILES: largest
fragment, canonical isomeric, flagged ``in_training``), approaching 1 when
the training set holds nothing similar.

The per-molecule summary ``training_distance`` is the 66th percentile of the
per-column distances: at least two-thirds of the columns have a training
molecule this close or closer. Per-column distances and the nearest training
molecules (keys, similarities, labels) go to
``TrainingDistanceRunResult.details``.
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

from eosquality.scores._base import ScoreComponent, read_json
from eosquality.scores._helpers import _standardize
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState
from eosquality.utils.logging import logger

SUBFOLDER = "training_distance"
STATE_FILE = "state.json"
# Quantile across columns for the per-molecule summary (the distance analogue
# of the Q66 aggregate typicality and extremity use over features).
SUMMARY_QUANTILE = 0.66
# Nearest training neighbours reported per (molecule, column).
N_NEIGHBORS = 5

DETAIL_COLUMNS = [
    "key",
    "column",
    "distance",
    "n_train",
    "in_training",
    "nn_keys",
    "nn_similarities",
    "nn_y",
]


@dataclass
class TrainingDistanceRunResult:
    """Result returned by :meth:`TrainingDistance.run`."""

    score: pd.Series  # (n_query,) summary distance across columns, in [0, 1]
    n_columns: pd.Series  # (n_query,) columns contributing to the summary
    in_training_any: pd.Series  # (n_query,) query is a training molecule of a column
    details: pd.DataFrame  # one row per (query, column)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingDistance(ScoreComponent):
    """Per-column Morgan distance to the nearest training molecule."""

    NAME = SUBFOLDER
    USES_SHARED = False  # X only: needs the training sets, not the reference
    USES_TRAINING = True

    def __init__(self) -> None:
        super().__init__()
        self._columns: list[str] | None = None

    def fit(
        self, *, training: TrainingFitState, shared: SharedFitState | None = None
    ) -> TrainingDistance:
        """Attach the training sets; there is nothing to calibrate.

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
            Summary distance, per-column details and run metadata.
        """
        self._check_fitted()
        assert self._training is not None
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
        distance = np.full((len(query), len(names)), np.nan)
        in_train = np.zeros((len(query), len(names)), dtype=bool)
        details = []
        for j, name in enumerate(names):
            column = self._training.columns[name]
            sims, nn_idx, hit = _nearest_training(
                self._training.indices[name],
                [std[i] for i in rows],
                min(N_NEIGHBORS, column.n),
            )
            distance[rows, j] = 1.0 - sims[:, 0] if len(rows) else []
            in_train[rows, j] = hit
            details.append(
                _details_rows([keys[i] for i in rows], column, sims, nn_idx, hit)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
            summary = np.nanquantile(distance, SUMMARY_QUANTILE, axis=1)
        return TrainingDistanceRunResult(
            score=pd.Series(summary, index=idx, name="training_distance"),
            n_columns=pd.Series(
                np.isfinite(distance).sum(axis=1), index=idx, name="training_n_columns"
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
                "summary_quantile": SUMMARY_QUANTILE,
            },
        )

    def _save_own(self, folder: pathlib.Path) -> None:
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"columns": self._columns}, f, indent=2)

    def _load_own(self, folder: pathlib.Path) -> None:
        assert self._training is not None
        names = read_json(folder / STATE_FILE, self.NAME)["columns"]
        if names != self._training.column_names:
            raise ValueError(
                "training_distance/state.json columns do not match "
                "training_sets/metadata.json."
            )
        self._columns = names

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return self._training is not None and self._columns is not None

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


def _details_rows(keys, column, sims, nn_idx, hit) -> pd.DataFrame:
    """One details row per query for one training column."""
    ids = column.ids
    return pd.DataFrame(
        {
            "key": keys,
            "column": column.name,
            "distance": 1.0 - sims[:, 0] if len(sims) else [],
            "n_train": column.n,
            "in_training": hit,
            "nn_keys": ["|".join(ids[t] for t in row) for row in nn_idx],
            "nn_similarities": ["|".join(f"{v:.3f}" for v in row) for row in sims],
            "nn_y": [
                "|".join(_fmt(column.y[t]) for t in row) if column.y is not None else ""
                for row in nn_idx
            ],
        },
        columns=DETAIL_COLUMNS,
    )


def _nearest_training(vi, query_smiles: list[str], k: int):
    """Top-k training neighbours per query (closest first), self included.

    Returns ``(similarities (n, k), indices (n, k), in_training (n,))``, where
    ``in_training`` marks queries whose standardised SMILES is a training
    molecule of this column.
    """
    if not query_smiles:
        return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64), np.zeros(0, bool)
    dist, nn = vi.query(query_smiles, k=k, show_progress=False)
    members = set(vi.smiles)
    hit = np.array([smi in members for smi in query_smiles])
    return (1.0 - dist).astype(np.float64), nn, hit


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"
