"""Physicochemical applicability domain against the model's training sets.

Training modality, X only. Per output column, the mean Euclidean distance
from the query to its 5 nearest training molecules over RDKit physchem
descriptors scaled with the reference library's scaler (clipped to ±10;
:mod:`eosquality.scores._physchem_domain`). Two columns come from it, both
higher-is-closer: ``trn_physchem_pct``, 1 − the mid-rank percentile of the
distance among the training molecules' own leave-one-out distances (~0.5 for
"as ordinary as a typical training molecule", near 0 for "further out than
almost all of them", summarised across columns at the 66th percentile of the
distance, as for training similarity), and ``trn_physchem_raw``, the similarity
``1 - d / PAIR_MEDIAN`` (0 for a random library pair, unclipped). The distance
itself is ``trn_physchem_dist``, in the details file.

Deliberately the same method as ``trn_tanimoto``, in a different space.
``trn_tanimoto`` asks whether the query resembles a *specific* training
molecule (Tanimoto over Morgan fingerprints); this asks whether its bulk
properties fall where the training set's do. OECD guidance asks for both
rather than a choice between them - "the AD should define the structural,
physicochemical and response space of the model" (ENV/JM/MONO(2007)2 §3.1
¶99), and "the different AD methods should not be seen as in competition
with one another" (¶128).

They are related but not redundant: on eos4e40 the two correlate at about
+0.5, so a molecule can be structurally novel while physicochemically
ordinary, or the reverse.
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
from eosquality.scores._helpers import _cdf_score
from eosquality.scores._physchem_domain import PhyschemDomain
from eosquality.scores._training_helpers import (
    SUMMARY_QUANTILE,
    TrainingQuery,
    _columns_summary,
)
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState
from eosquality.utils import console
from eosquality.utils.logging import logger

SUBFOLDER = "training_physchem"
STATE_FILE = "state.json"


@dataclass
class TrainingPhyschemRunResult:
    """Result returned by :meth:`TrainingPhyschem.run`."""

    score: pd.Series  # (n_query,) similarity percentile across columns, in (0, 1)
    score_raw: pd.Series  # (n_query,) similarity 1 - d / pair median, unclipped
    distance: pd.Series  # (n_query,) mean distance to the k nearest, library SD units
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingPhyschem(ScoreComponent):
    """Per-column physchem k-NN distance against the training sets."""

    NAME = SUBFOLDER
    USES_SHARED = False  # X only: needs the training sets, not the reference
    USES_TRAINING = True

    def __init__(self) -> None:
        super().__init__()
        self._domains: dict[str, PhyschemDomain] | None = None

    def fit(
        self, *, training: TrainingFitState, shared: SharedFitState | None = None
    ) -> TrainingPhyschem:
        """Fit one physchem domain per training column.

        Parameters
        ----------
        training : TrainingFitState
            Training sets and their per-column indices.
        shared : SharedFitState, optional
            Unused; accepted for a uniform component interface.

        Returns
        -------
        TrainingPhyschem
            ``self``, fitted.
        """
        from eosquality.library.physchem import (
            canonical_scaler,
            compute_physchem_raw,
        )

        scaler = canonical_scaler()
        t0 = time.perf_counter()
        self._training = training
        self._domains = {}
        names = training.column_names
        # The columns of one panel share most molecules: describe each once.
        union = sorted({s for n in names for s in training.columns[n].smiles})
        row_of = {smi: i for i, smi in enumerate(union)}
        raw_all = compute_physchem_raw(
            union, show_progress=True, label="physchem descriptors"
        )
        by_set: dict[str, PhyschemDomain] = {}  # a panel's columns share a domain
        for name in console.track(names, "physchem, columns"):
            column = training.columns[name]
            if column.signature not in by_set:
                rows = [row_of[smi] for smi in column.smiles]
                by_set[column.signature] = PhyschemDomain.fit(raw_all[rows], scaler)
            self._domains[name] = by_set[column.signature]
            logger.debug(
                f"physchem domain | column {name!r}: {column.n:,} molecules, "
                f"k={self._domains[name].k}"
            )
        self._finish_fit(t0)
        return self

    def run(
        self, query: pd.DataFrame, features: TrainingQuery | None = None
    ) -> TrainingPhyschemRunResult:
        """Physchem distance of each query to every column's training set.

        Parameters
        ----------
        query : pandas.DataFrame
            Needs an ``input`` SMILES column.
        features : TrainingQuery, optional
            The query's features, shared with the other training scores.

        Returns
        -------
        TrainingPhyschemRunResult
        """
        self._check_fitted()
        assert self._domains is not None and self._training is not None
        if features is None:
            features = TrainingQuery.from_frame(query)
        rows, names = features.rows, list(self._domains)
        idx = list(query.index)
        shape = (len(query), len(names))
        raw = np.full(shape, np.nan)
        calibrated = np.full(shape, np.nan)
        if len(features.smiles):
            physchem = features.physchem
            measured: dict[int, np.ndarray] = {}  # columns sharing a domain
            for j, name in enumerate(names):
                domain = self._domains[name]
                if id(domain) not in measured:
                    measured[id(domain)] = domain.measure(
                        physchem, self._training.columns[name].rows_of(features.smiles)
                    )
                distance = measured[id(domain)]
                raw[rows, j] = distance
                calibrated[rows, j] = _cdf_score(distance, domain.sorted_distances)
        summary_distance = _columns_summary(raw)
        return TrainingPhyschemRunResult(
            score=pd.Series(
                1.0 - _columns_summary(calibrated),
                index=idx,
                name="trn_physchem_pct",
            ),
            score_raw=pd.Series(
                next(iter(self._domains.values())).similarity(summary_distance),
                index=idx,
                name="trn_physchem_raw",
            ),
            distance=pd.Series(summary_distance, index=idx, name="trn_physchem_dist"),
            metadata={
                "columns": names,
                "k": {n: d.k for n, d in self._domains.items()},
                "summary_quantile": SUMMARY_QUANTILE,
            },
        )

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._domains is not None
        saved: dict[int, str] = {}  # id of a domain → its folder
        folders: dict[str, str] = {}
        for name, domain in self._domains.items():
            if id(domain) not in saved:
                saved[id(domain)] = f"c{len(saved):03d}"
                domain.save(folder / saved[id(domain)])
            folders[name] = saved[id(domain)]
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"columns": list(self._domains), "folders": folders}, f, indent=2)

    def _load_own(self, folder: pathlib.Path) -> None:
        assert self._training is not None
        state = read_json(folder / STATE_FILE, self.NAME)
        names = state["columns"]
        unknown = set(names) - set(self._training.column_names)
        if unknown:
            raise ValueError(
                f"{SUBFOLDER}/state.json has columns {sorted(unknown)} that are not "
                "in training_sets/metadata.json."
            )
        loaded: dict[str, PhyschemDomain] = {}
        for sub in dict.fromkeys(state["folders"][name] for name in names):
            loaded[sub] = PhyschemDomain.load(folder / sub)
        self._domains = {name: loaded[state["folders"][name]] for name in names}

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return self._training is not None and self._domains is not None

    @property
    def fit_summary(self) -> str:
        """One line for the fit log.

        Returns
        -------
        str
        """
        sizes = sorted({d.n_train for d in self.domains_.values()})
        span = f"{sizes[0]:,}-{sizes[-1]:,}" if len(sizes) > 1 else f"{sizes[0]:,}"
        return f"{len(self.domains_)} physchem domain(s) · {span} molecules"

    @property
    def domains_(self) -> dict[str, PhyschemDomain]:
        """The per-column fitted domains.

        Returns
        -------
        dict of str to PhyschemDomain
        """
        self._check_fitted()
        assert self._domains is not None
        return self._domains
