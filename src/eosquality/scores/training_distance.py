"""Training distance: how far is the query from the model's training molecules?

Training mode, X only. One value per molecule for the whole model, built
from the output columns' training sets without pooling them:

- **Per column (internal).** The raw distance is ``1 − mean Tanimoto
  similarity`` (Morgan, radius 2, 2048 bits) to the query's **k = 5 nearest
  training molecules** (Sheridan et al. 2004). A query that is itself a
  training molecule (same standardised SMILES: largest fragment, canonical
  isomeric) drops its own entry, so it gets its leave-one-out value. The
  calibrated distance is the mid-rank CDF of the raw value against the
  column's own leave-one-out raw values (each training molecule vs its k
  nearest *other* training molecules): ~0.5 for a query as close as a typical
  training molecule, near 1 when farther than almost all of them.
- **Whole model.** Both published columns are similarities, higher is closer.
  ``trn_tanimoto_pct`` is 1 − the 66th percentile across columns of the
  calibrated distance (the similarity percentile: ~0.5 for a query as close as
  a typical training molecule, near 0 when farther than almost all of them),
  and ``trn_tanimoto_raw`` is the mean Tanimoto similarity at the same point
  (1 − the 66th percentile of the raw distances): at least two-thirds of the
  columns are this close or closer. Calibrated
  values are percentiles of each column's own training set, so columns of
  very different sizes and densities combine fairly; a large training set
  cannot hide that the query is far from a small one.

Higher is closer; there is no in/out cutoff. ``trn_in_training`` flags queries
that are a training molecule of any column. The details table has one row
per query with the 5 nearest training molecules over all columns (keys,
similarities, the columns each belongs to).
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import _cdf_score, _sorted_finite
from eosquality.scores._training_helpers import (
    SUMMARY_QUANTILE,
    TrainingQuery,
    _columns_summary,
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

# The training details table, one row per query. ``trn_difficulty`` is
# inserted after ``trn_tanimoto_raw`` when fitted. ``nn_*`` describe the
# K_NEIGHBORS nearest training molecules over all columns, closest first,
# "|"-separated (``nn_columns``: ";" between the columns of one molecule).
DETAIL_COLUMNS = [
    "key",
    "input",
    "trn_tanimoto_pct",
    "trn_tanimoto_raw",
    "trn_in_training",
    "nn1_similarity",
    "nn_smiles",
    "nn_keys",
    "nn_similarities",
    "nn_columns",
]


@dataclass
class TrainingDistanceRunResult:
    """Result returned by :meth:`TrainingDistance.run`."""

    score: pd.Series  # (n_query,) similarity percentile across columns, in (0, 1)
    score_raw: pd.Series  # (n_query,) mean Tanimoto similarity to the k nearest, [0, 1]
    in_training: pd.Series  # (n_query,) query is a training molecule of a column
    details: pd.DataFrame  # one row per query (DETAIL_COLUMNS)
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

    def run(
        self, query: pd.DataFrame, features: TrainingQuery | None = None
    ) -> TrainingDistanceRunResult:
        """Distance of each query to every column's training set.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column; ``key`` (if present) labels the
            rows of the details table.
        features : TrainingQuery, optional
            The query's features, shared with training difficulty (built from
            ``query`` when omitted).

        Returns
        -------
        TrainingDistanceRunResult
            Whole-model calibrated and raw distances, details and metadata.
        """
        self._check_fitted()
        assert self._training is not None and self._k is not None
        assert self._loo is not None
        if features is None:
            features = TrainingQuery.from_frame(query)
        idx = list(query.index)
        keys = (
            query["key"].astype(str).tolist()
            if "key" in query.columns
            else [str(i) for i in idx]
        )
        rows = features.rows
        names = self._training.column_names
        raw = np.full((len(query), len(names)), np.nan)
        calibrated = np.full_like(raw, np.nan)
        in_train = np.zeros(raw.shape, dtype=bool)
        raw[rows], calibrated[rows], in_train[rows], neighbours = self._per_column(
            features
        )
        # Per-column calibration is a distance percentile (higher = farther); the
        # published score is its similarity: 1 minus the Q66 is the Q34 of
        # the similarity percentiles, i.e. at least two-thirds of the columns
        # are this close or closer.
        score = 1.0 - _columns_summary(calibrated)
        score_raw = 1.0 - _columns_summary(raw)  # mean Tanimoto similarity
        details = pd.DataFrame(
            {
                "key": keys,
                "input": query["input"].tolist(),
                "trn_tanimoto_pct": score,
                "trn_tanimoto_raw": score_raw,
                "trn_in_training": in_train.any(axis=1),
            },
            index=idx,
        )
        neighbour_columns = _neighbours(
            [self._training.columns[n] for n in names], neighbours
        )
        for name, values in neighbour_columns.items():
            details[name] = pd.Series(
                values, index=[idx[i] for i in rows], dtype=object
            )
        details = details[DETAIL_COLUMNS].reset_index(drop=True)
        details["nn1_similarity"] = details["nn1_similarity"].astype(float)
        return TrainingDistanceRunResult(
            score=pd.Series(score, index=idx, name="trn_tanimoto_pct"),
            score_raw=pd.Series(score_raw, index=idx, name="trn_tanimoto_raw"),
            in_training=pd.Series(
                in_train.any(axis=1), index=idx, name="trn_in_training"
            ),
            details=details,
            metadata={
                "n_columns": len(names),
                "columns": names,
                "k": K_NEIGHBORS,
                "summary_quantile": SUMMARY_QUANTILE,
            },
        )

    def _per_column(self, features: TrainingQuery | list[str]):
        """Per-column raw and calibrated distances for standardised SMILES.

        Returns ``(raw (n, n_columns), calibrated (n, n_columns),
        in_training (n, n_columns), neighbours)``, where ``neighbours`` holds
        one ``(distances (n, k), training indices (n, k))`` pair per column.
        """
        assert self._training is not None and self._k is not None
        assert self._loo is not None
        if not isinstance(features, TrainingQuery):
            features = TrainingQuery(features)
        names = self._training.column_names
        raw = np.full((len(features.smiles), len(names)), np.nan)
        calibrated = np.full_like(raw, np.nan)
        in_train = np.zeros(raw.shape, dtype=bool)
        neighbours = []
        for j, name in enumerate(names):
            dist, nn_idx, hit = features.nearest(
                self._training.indices[name], self._k[name]
            )
            raw[:, j] = dist.mean(axis=1)
            calibrated[:, j] = _cdf_score(
                raw[:, j], self._loo[name], higher_is_higher=True
            )
            in_train[:, j] = hit
            neighbours.append((dist, nn_idx))
        return raw, calibrated, in_train, neighbours

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


def _neighbours(columns, neighbours) -> dict[str, list]:
    """The ``nn_*`` details columns for the valid queries, in row order.

    Each column's nearest training molecules are pooled, deduplicated by
    standardised SMILES (a molecule in several training sets keeps the key of
    its first column and lists every column it belongs to) and the
    ``K_NEIGHBORS`` closest are kept.
    """
    out: dict[str, list] = {
        "nn1_similarity": [],
        "nn_smiles": [],
        "nn_keys": [],
        "nn_similarities": [],
        "nn_columns": [],
    }
    n_rows = len(neighbours[0][0]) if neighbours else 0
    for r in range(n_rows):
        best: dict[str, tuple[float, str, list[str]]] = {}
        for column, (dist, nn_idx) in zip(columns, neighbours, strict=True):
            for d, t in zip(dist[r], nn_idx[r], strict=True):
                smi = column.smiles[t]
                if smi in best:
                    best[smi][2].append(column.name)
                else:
                    best[smi] = (float(d), column.ids[t], [column.name])
        top = sorted(best.items(), key=lambda item: item[1][0])[:K_NEIGHBORS]
        out["nn1_similarity"].append(1 - top[0][1][0] if top else np.nan)
        out["nn_smiles"].append("|".join(smi for smi, _ in top))
        out["nn_keys"].append("|".join(v[1] for _, v in top))
        out["nn_similarities"].append("|".join(f"{1 - v[0]:.3f}" for _, v in top))
        out["nn_columns"].append("|".join(";".join(v[2]) for _, v in top))
    return out
