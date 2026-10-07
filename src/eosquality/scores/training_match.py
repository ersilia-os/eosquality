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

from eosquality.scores._base import ScoreComponent, require_file
from eosquality.scores._training_helpers import TrainingQuery
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState
from eosquality.utils import console

SUBFOLDER = "training_match"
KEYS_FILE = "connectivity_keys.npz"
# Characters of an InChIKey that form its connectivity layer (first block).
CONNECTIVITY_LENGTH = 14


@dataclass
class TrainingMatchRunResult:
    """Result returned by :meth:`TrainingMatch.run`."""

    match: pd.Series  # (n_query,) Int64: 1 / 0, NA for unparsable rows
    scaffold: pd.Series  # (n_query,) Int64: 1 / 0, NA for no scaffold
    metadata: dict[str, Any] = field(default_factory=dict)


def connectivity_layer(smiles: str) -> str:
    """InChIKey connectivity layer of a SMILES; ``""`` when it has none.

    Parameters
    ----------
    smiles : str
        A SMILES string.

    Returns
    -------
    str
        The first 14 characters of the InChIKey, or ``""`` for an empty or
        unparsable structure.
    """
    from rdkit import Chem, rdBase

    if not smiles:
        return ""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            return ""
        key = Chem.MolToInchiKey(mol)
    return key[:CONNECTIVITY_LENGTH]


def _layers(
    smiles: list[str], label: str = "InChIKey layers"
) -> tuple[np.ndarray, np.ndarray]:
    """Connectivity layers of the molecules and of their Murcko scaffolds.

    Parameters
    ----------
    smiles : list of str
        Standardised SMILES.
    label : str, optional
        Progress-bar title.

    Returns
    -------
    tuple of numpy.ndarray
        ``(molecule layers, scaffold layers)``, both length ``len(smiles)``;
        ``""`` where there is none.
    """
    from eosquality.training.folds import _scaffold

    molecules = np.empty(len(smiles), dtype=object)
    scaffolds = np.empty(len(smiles), dtype=object)
    with console.progress(label) as bar:
        task = bar.add_task(label, total=len(smiles))
        for i, s in enumerate(smiles):
            molecules[i] = connectivity_layer(s)
            scaffolds[i] = connectivity_layer(_scaffold(s))
            bar.advance(task)
    return molecules, scaffolds


def _flags(layers: np.ndarray, known: np.ndarray) -> pd.Series:
    """1 / 0 for layers found in ``known``; NA where the layer is empty."""
    present = np.array([bool(x) for x in layers], dtype=bool)
    found = np.isin(layers, known)
    return pd.Series(np.where(present, found.astype(int), 0), dtype="Int64").mask(
        ~present
    )


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
        self._molecules = np.unique(molecules[molecules != ""]).astype(str)
        self._scaffolds = np.unique(scaffolds[scaffolds != ""]).astype(str)
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
                "n_molecules": len(self._molecules),
                "n_scaffolds": len(self._scaffolds),
            },
        )

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._molecules is not None and self._scaffolds is not None
        np.savez(
            folder / KEYS_FILE, molecules=self._molecules, scaffolds=self._scaffolds
        )

    def _load_own(self, folder: pathlib.Path) -> None:
        path = require_file(folder / KEYS_FILE, self.NAME)
        with np.load(path, allow_pickle=False) as arrays:
            self._molecules = arrays["molecules"]
            self._scaffolds = arrays["scaffolds"]

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return self._molecules is not None and self._scaffolds is not None
