"""Training difficulty: how hard is the query to predict, judging by the training data?

Training mode, needs labels ``y``. One value per molecule for the whole
model. For each output column with at least ``MIN_LABELLED`` labels, an
**error model** learns where in chemical space the endpoint is hard to
predict, following UNIQUE's error models (see
:mod:`eosquality.scores._error_model`): a surrogate random forest fitted
with scaffold-grouped CV gives out-of-fold residuals, and a second random
forest predicts them from UNIQUE's feature set (i): MACCS keys, base UQ
metrics (kNN distance, KDE densities, ensemble variance) and the prediction.

The query's predicted error is calibrated as its percentile among the
training molecules' out-of-fold predicted errors: ~0.5 is as hard as a
typical training molecule, near 1 among the hardest. ``trn_difficulty``
is the 66th percentile across labelled columns. There is no raw column:
predicted errors are in each endpoint's own units and cannot be combined.

This is a **data** difficulty rank, specific to the endpoint, not the
deployed model's error; per-column Spearman correlations between predicted
and actual out-of-fold errors are reported in the metadata so the ranking
can be trusted, or not.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.scores._base import ScoreComponent, read_json
from eosquality.scores._error_model import (
    MIN_LABELLED,
    EndpointErrorModel,
    fit_endpoint,
    is_eligible,
)
from eosquality.scores._helpers import _cdf_score
from eosquality.scores._training_helpers import (
    SUMMARY_QUANTILE,
    TrainingQuery,
    _columns_summary,
)
from eosquality.scores.training_distance import K_NEIGHBORS
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState
from eosquality.utils.logging import logger

# Out-of-fold Spearman below which a column's error model is reported as
# weak: it ranks its own held-out errors barely better than chance (seen on
# noisy, chemically diverse endpoints such as aqueous solubility).
WEAK_SPEARMAN = 0.2

SUBFOLDER = "training_difficulty"
STATE_FILE = "state.json"


@dataclass
class TrainingDifficultyRunResult:
    """Result returned by :meth:`TrainingDifficulty.run`."""

    score: pd.Series  # (n_query,) calibrated whole-model difficulty, in (0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingDifficulty(ScoreComponent):
    """Whole-model difficulty from per-column learned error models."""

    NAME = SUBFOLDER
    USES_SHARED = False  # needs the training sets, not the reference
    USES_TRAINING = True

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[str, EndpointErrorModel] | None = None

    @staticmethod
    def can_fit(training: TrainingFitState) -> bool:
        """Whether any column has enough labels for an error model.

        Parameters
        ----------
        training : TrainingFitState
            Training sets.

        Returns
        -------
        bool
        """
        return any(is_eligible(c) for c in training.columns.values())

    def fit(
        self, *, training: TrainingFitState, shared: SharedFitState | None = None
    ) -> TrainingDifficulty:
        """Fit one surrogate + error model per labelled column.

        Parameters
        ----------
        training : TrainingFitState
            Training sets and their per-column Morgan indices.
        shared : SharedFitState, optional
            Unused; accepted for a uniform component interface.

        Returns
        -------
        TrainingDifficulty
            ``self``, fitted.

        Raises
        ------
        ValueError
            If no column has at least ``MIN_LABELLED`` labels.
        """
        if not self.can_fit(training):
            raise ValueError(
                f"TrainingDifficulty needs a column with ≥ {MIN_LABELLED} labels."
            )
        t0 = time.perf_counter()
        self._training = training
        self._models = {}
        for name, column in training.columns.items():
            if not is_eligible(column):
                continue
            t = time.perf_counter()
            k = min(K_NEIGHBORS, column.n - 2)
            model = fit_endpoint(column, training.indices[name], k)
            self._models[name] = model
            logger.info(
                f"training difficulty | column {name!r}: n={model.n_labelled:,} | "
                f"folds={model.cv} | Spearman(predicted, OOF error)="
                f"{model.spearman:.3f} | {time.perf_counter() - t:.1f}s"
            )
            _warn_if_degraded(name, model)
        self._finish_fit(t0)
        return self

    def run(
        self, query: pd.DataFrame, features: TrainingQuery | None = None
    ) -> TrainingDifficultyRunResult:
        """Whole-model difficulty of each query.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column.
        features : TrainingQuery, optional
            The query's features, shared with training distance (built from
            ``query`` when omitted).

        Returns
        -------
        TrainingDifficultyRunResult
            Calibrated whole-model difficulty and per-column metadata.
        """
        self._check_fitted()
        assert self._training is not None and self._models is not None
        if features is None:
            features = TrainingQuery.from_frame(query)
        rows = features.rows
        names = list(self._models)
        calibrated = np.full((len(query), len(names)), np.nan)
        for j, name in enumerate(names):
            model = self._models[name]
            predicted = model.predict(
                self._training.columns[name], self._training.indices[name], features
            )
            calibrated[rows, j] = _cdf_score(
                predicted, model.sorted_oof_error, higher_is_higher=True
            )
        return TrainingDifficultyRunResult(
            score=pd.Series(
                _columns_summary(calibrated),
                index=list(query.index),
                name="trn_difficulty",
            ),
            metadata={
                "columns": names,
                "spearman": {n: m.spearman for n, m in self._models.items()},
                "n_labelled": {n: m.n_labelled for n, m in self._models.items()},
                "cv": {n: m.cv for n, m in self._models.items()},
                "summary_quantile": SUMMARY_QUANTILE,
            },
        )

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._models is not None
        names = list(self._models)
        for j, name in enumerate(names):
            self._models[name].save(folder / f"c{j:03d}")
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"columns": names}, f, indent=2)

    def _load_own(self, folder: pathlib.Path) -> None:
        assert self._training is not None
        names = read_json(folder / STATE_FILE, self.NAME)["columns"]
        unknown = set(names) - set(self._training.column_names)
        if unknown:
            raise ValueError(
                f"training_difficulty/state.json names columns {sorted(unknown)} "
                "that are not in training_sets/metadata.json."
            )
        self._models = {
            name: EndpointErrorModel.load(folder / f"c{j:03d}", self.NAME)
            for j, name in enumerate(names)
        }

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return self._training is not None and self._models is not None

    @property
    def models_(self) -> dict[str, EndpointErrorModel]:
        """Per-column fitted error models.

        Returns
        -------
        dict of str to EndpointErrorModel
        """
        self._check_fitted()
        assert self._models is not None
        return self._models


def _warn_if_degraded(name: str, model: EndpointErrorModel) -> None:
    """Warn when a column's error model is less trustworthy than usual."""
    if model.cv == "random":
        logger.warning(
            f"training difficulty | column {name!r}: too few or too unbalanced "
            "Murcko scaffolds for scaffold folds; random folds were used, so its "
            "out-of-fold errors (and Spearman) are optimistic."
        )
    if not np.isfinite(model.spearman):
        logger.warning(
            f"training difficulty | column {name!r}: the error model could not be "
            "validated (constant out-of-fold errors); its ranking is uninformative."
        )
    elif model.spearman < WEAK_SPEARMAN:
        logger.warning(
            f"training difficulty | column {name!r}: out-of-fold Spearman "
            f"{model.spearman:.2f} (< {WEAK_SPEARMAN}); this column's errors are "
            "barely predictable from its training data, so it adds little to "
            "trn_difficulty."
        )
