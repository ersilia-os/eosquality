"""Exact-structure and scaffold match against the model's training sets.

Training modality, X only. Two 1/0 flags per query, over the union of every
output column's training molecules:

- ``trn_match``: 1 when the query's InChIKey **connectivity layer** (the first
  14 characters, the hash of the heavy-atom skeleton and hydrogens) equals that
  of a training molecule. It is the same compound ignoring stereochemistry,
  isotopes, charge state and mobile-hydrogen tautomerism, so it catches ionised
  forms, enantiomers and tautomers that a SMILES comparison would miss.
- ``trn_scaffold``: 1 when the connectivity layer of the query's Murcko
  scaffold equals that of some training molecule's scaffold. It is missing
  (``NA``, an empty CSV cell) for a query with no scaffold, such as an
  acyclic molecule, because the question has no answer there, and is not 0.

Both are missing for an unparsable query. Unlike ``trn_tanimoto`` and
``trn_physchem`` they are exact set lookups, with no calibration.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.scores._base import ScoreComponent
from eosquality.scores._match_keys import (
    KEYS_FILE,
    _flags,
    _layers,
    _scaffold,  # noqa: F401  (re-exported: the scaffold helper used by the tests)
    connectivity_layer,  # noqa: F401  (re-exported)
    load_keys,
    save_keys,
    unique_keys,
)
from eosquality.scores._training_helpers import TrainingQuery
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState

SUBFOLDER = "training_match"


@dataclass
class TrainingMatchRunResult:
    """Result returned by :meth:`TrainingMatch.run`."""

    match: pd.Series  # (n_query,) Int64: 1 / 0, NA for unparsable rows
    scaffold: pd.Series  # (n_query,) Int64: 1 / 0, NA for no scaffold
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingMatch(ScoreComponent):
    """Connectivity-layer match of a query against all training molecules."""

    NAME = SUBFOLDER
    USES_SHARED = False  # X only: needs the training sets, not the reference
    USES_TRAINING = True

    def __init__(self) -> None:
        super().__init__()
        self._molecules: np.ndarray | None = None  # sorted unique layers
        self._scaffolds: np.ndarray | None = None

    def fit(
        self, *, training: TrainingFitState, shared: SharedFitState | None = None
    ) -> TrainingMatch:
        """Collect the connectivity layers of every training molecule.

        Parameters
        ----------
        training : TrainingFitState
            Training sets of every output column.
        shared : SharedFitState, optional
            Unused; accepted for a uniform component interface.

        Returns
        -------
        TrainingMatch
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        self._training = training
        smiles = sorted(
            {s for n in training.column_names for s in training.columns[n].smiles}
        )
        molecules, scaffolds = _layers(smiles)
        self._molecules, self._scaffolds = (
            unique_keys(molecules),
            unique_keys(scaffolds),
        )
        self._finish_fit(t0)
        return self

    def run(
        self, query: pd.DataFrame, features: TrainingQuery | None = None
    ) -> TrainingMatchRunResult:
        """Flag each query that matches a training molecule or scaffold.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column.
        features : TrainingQuery, optional
            The query's standardised SMILES, shared with the other training
            scores (built from ``query`` when omitted).

        Returns
        -------
        TrainingMatchRunResult
        """
        self._check_fitted()
        assert self._molecules is not None and self._scaffolds is not None
        if features is None:
            features = TrainingQuery.from_frame(query)
        molecules, scaffolds = _layers(features.smiles, "query InChIKey layers")
        idx = list(query.index)
        match = pd.Series(pd.NA, index=range(len(idx)), dtype="Int64")
        scaffold = match.copy()
        rows = np.asarray(features.rows, dtype=int)
        match.iloc[rows] = _flags(molecules, self._molecules).to_numpy()
        scaffold.iloc[rows] = _flags(scaffolds, self._scaffolds).to_numpy()
        match.index = scaffold.index = idx
        match.name, scaffold.name = "trn_match", "trn_scaffold"
        return TrainingMatchRunResult(
            match=match,
            scaffold=scaffold,
            metadata={
                "n_molecules": self.n_molecules,
                "n_scaffolds": self.n_scaffolds,
            },
        )

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._molecules is not None and self._scaffolds is not None
        save_keys(folder / KEYS_FILE, self._molecules, self._scaffolds)

    def _load_own(self, folder: pathlib.Path) -> None:
        self._molecules, self._scaffolds = load_keys(folder / KEYS_FILE, self.NAME)

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return self._molecules is not None and self._scaffolds is not None

    @property
    def n_molecules(self) -> int:
        """Number of distinct training connectivity layers.

        Returns
        -------
        int
        """
        self._check_fitted()
        assert self._molecules is not None
        return len(self._molecules)

    @property
    def n_scaffolds(self) -> int:
        """Number of distinct training scaffold connectivity layers.

        Returns
        -------
        int
        """
        self._check_fitted()
        assert self._scaffolds is not None
        return len(self._scaffolds)

    @property
    def fit_summary(self) -> str:
        """One line for the fit log.

        Returns
        -------
        str
        """
        return f"{self.n_molecules:,} structures · {self.n_scaffolds:,} scaffolds"
