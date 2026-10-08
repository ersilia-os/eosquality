"""Exact-structure and scaffold match against the reference library.

Reference modality, X only: the molecules of the **reference library** (about
1.35M), not the model's behaviour on them. Two 1/0 flags per query:

- ``ref_match``: 1 when the query's InChIKey **connectivity layer** (the first
  14 characters, the hash of the heavy-atom skeleton and hydrogens) equals that
  of a library molecule: the same compound ignoring stereochemistry, isotopes,
  charge state and mobile-hydrogen tautomerism.
- ``ref_scaffold``: 1 when the connectivity layer of the query's Murcko
  scaffold equals that of some library molecule's scaffold. It is missing
  (``NA``, an empty CSV cell) for a query with no scaffold, such as an acyclic
  molecule, because the question has no answer there, and is not 0.

Both are missing for an unparsable query. They are exact set lookups with no
calibration, and they do not depend on the model: the keys are computed once
per library (``eosquality build``) and live in the library folder, not in each
model's artifacts. The artifact records only which library it was fitted
against; the keys are read from that library when the score first runs.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.library.reference import ReferenceLibrary
from eosquality.scores._base import ScoreComponent, read_json
from eosquality.scores._helpers import _standardize_all
from eosquality.scores._match_keys import _flags, _layers
from eosquality.shared.state import SharedFitState

SUBFOLDER = "match"
STATE_FILE = "state.json"


@dataclass
class ReferenceMatchRunResult:
    """Result returned by :meth:`ReferenceMatch.run`."""

    match: pd.Series  # (n_query,) Int64: 1 / 0, NA for unparsable rows
    scaffold: pd.Series  # (n_query,) Int64: 1 / 0, NA for no scaffold
    metadata: dict[str, Any] = field(default_factory=dict)


class ReferenceMatch(ScoreComponent):
    """Connectivity-layer match of a query against the reference library."""

    NAME = SUBFOLDER

    def __init__(self) -> None:
        super().__init__()
        self._library: ReferenceLibrary | None = None
        self._n_molecules: int | None = None
        self._n_scaffolds: int | None = None

    def fit(
        self,
        reference: pd.DataFrame | None = None,
        *,
        shared: SharedFitState,
        library: ReferenceLibrary,
    ) -> ReferenceMatch:
        """Bind to a reference library (its keys are computed by ``build``).

        Parameters
        ----------
        reference : pandas.DataFrame, optional
            Unused; accepted for a uniform component interface.
        shared : SharedFitState
            Shared state (it records the library the artifacts belong to).
        library : ReferenceLibrary
            The library whose match keys are used.

        Returns
        -------
        ReferenceMatch
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        self._shared = shared
        self._library = library
        molecules, scaffolds = library.match_keys()
        self._n_molecules, self._n_scaffolds = len(molecules), len(scaffolds)
        self._finish_fit(t0)
        return self

    def run(self, query: pd.DataFrame) -> ReferenceMatchRunResult:
        """Flag each query that matches a library molecule or scaffold.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column.

        Returns
        -------
        ReferenceMatchRunResult
        """
        self._check_fitted()
        molecules_known, scaffolds_known = self._resolve_library().match_keys()
        smiles = _standardize_all(list(query["input"]))
        rows = np.array([i for i, s in enumerate(smiles) if s], dtype=int)
        standardised = [smiles[i] for i in rows]
        molecules, scaffolds = _layers(standardised, "query InChIKey layers")
        idx = list(query.index)
        match = pd.Series(pd.NA, index=range(len(idx)), dtype="Int64")
        scaffold = match.copy()
        match.iloc[rows] = _flags(molecules, molecules_known).to_numpy()
        scaffold.iloc[rows] = _flags(scaffolds, scaffolds_known).to_numpy()
        match.index = scaffold.index = idx
        match.name, scaffold.name = "ref_match", "ref_scaffold"
        return ReferenceMatchRunResult(
            match=match,
            scaffold=scaffold,
            metadata={
                "n_molecules": self.n_molecules,
                "n_scaffolds": self.n_scaffolds,
            },
        )

    def _resolve_library(self) -> ReferenceLibrary:
        """The library bound at fit time, or the one the artifacts name."""
        if self._library is None:
            assert self._shared is not None
            meta = self._shared.metadata
            self._library = ReferenceLibrary.load(meta.library_path or None)
            if not meta.library_path and self._library.library_name != meta.library_id:
                from eosquality.exceptions import IncompatibleArtifactsError

                raise IncompatibleArtifactsError(
                    f"Artifacts were fit against reference library "
                    f"{meta.library_id!r} but the library found is "
                    f"{self._library.library_name!r}. Install a compatible "
                    "eosquality release or refit."
                )
        return self._library

    def _save_own(self, folder: pathlib.Path) -> None:
        with open(folder / STATE_FILE, "w") as f:
            json.dump(
                {"n_molecules": self.n_molecules, "n_scaffolds": self.n_scaffolds}, f
            )

    def _load_own(self, folder: pathlib.Path) -> None:
        state = read_json(folder / STATE_FILE, self.NAME)
        self._n_molecules = int(state["n_molecules"])
        self._n_scaffolds = int(state["n_scaffolds"])

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return (
            self._shared is not None
            and self._n_molecules is not None
            and self._n_scaffolds is not None
        )

    @property
    def n_molecules(self) -> int:
        """Number of distinct library connectivity layers.

        Returns
        -------
        int
        """
        self._check_fitted()
        assert self._n_molecules is not None
        return self._n_molecules

    @property
    def n_scaffolds(self) -> int:
        """Number of distinct library scaffold connectivity layers.

        Returns
        -------
        int
        """
        self._check_fitted()
        assert self._n_scaffolds is not None
        return self._n_scaffolds

    @property
    def fit_summary(self) -> str:
        """One line for the fit log.

        Returns
        -------
        str
        """
        return f"{self.n_molecules:,} structures · {self.n_scaffolds:,} scaffolds"
