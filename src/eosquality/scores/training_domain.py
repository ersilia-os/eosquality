"""Training domain: is the query inside each output column's training chemical space?

Training modality, X only. For every output column with a training set, the
raw value is the Tanimoto similarity (Morgan, radius 2, 2048 bits) of the
query's **nearest training molecule**. It is calibrated against that
column's own leave-one-out nearest-training similarities (each training
molecule vs its closest *other* training molecule), so a query that is as
close to the training set as training molecules are to each other scores
~0.5, and training molecules themselves score ~Uniform(0, 1).

A query that is itself a training molecule (same standardised SMILES:
largest fragment, canonical isomeric) drops its own entry, so it gets its
leave-one-out value and is flagged ``in_training``.

The per-molecule summary ``training_domain`` is the 34th percentile of the
per-column calibrated values ("at least two-thirds of the columns are at
least this in-domain"); it is not re-calibrated. Per-column values and the
nearest training neighbours (keys, similarities, labels) go to
``TrainingDomainRunResult.details``.
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
from eosquality.scores._helpers import _cdf_score, _standardize
from eosquality.shared.state import SharedFitState
from eosquality.training.state import TrainingFitState
from eosquality.utils.logging import logger

SUBFOLDER = "training_domain"
STATE_FILE = "state.json"
SIMILARITIES_FILE = "loo_nearest_similarities.npz"
# Quantile across columns for the per-molecule summary (mirror of the
# AGGREGATE_QUANTILE = 0.66 used over features by typicality / extremity).
SUMMARY_QUANTILE = 0.34
# Nearest training neighbours reported per (molecule, column).
N_NEIGHBORS = 5


@dataclass
class TrainingDomainRunResult:
    """Result returned by :meth:`TrainingDomain.run`."""

    score: pd.Series  # (n_query,) summary across columns, in (0, 1]
    score_raw: pd.Series  # (n_query,) same quantile of the raw nearest similarity
    n_columns: pd.Series  # (n_query,) columns contributing to the summary
    in_training_any: pd.Series  # (n_query,) query is a training molecule of any column
    details: pd.DataFrame  # one row per (query, column)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingDomain(ScoreComponent):
    """Per-column applicability domain from training-set nearest analogues."""

    NAME = SUBFOLDER
    USES_SHARED = False  # X-only: needs the training sets, not the reference
    USES_TRAINING = True

    def __init__(self) -> None:
        super().__init__()
        self._loo: dict[str, np.ndarray] | None = None  # column → sorted LOO sims
        self._reference_domain: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self, *, training: TrainingFitState, shared: SharedFitState | None = None
    ) -> TrainingDomain:
        """Read each column's leave-one-out nearest similarities from its index."""
        t0 = time.perf_counter()
        loo, anchors = {}, {}
        for name, vi in training.indices.items():
            sims = 1.0 - vi.self_knn_distances(1)[:, 0].astype(np.float64)
            loo[name] = np.sort(sims)
            anchors[name] = float(
                np.mean(_cdf_score(sims, loo[name], higher_is_higher=True))
            )
        self._shared = shared
        self._training = training
        self._loo = loo
        self._reference_domain = anchors
        self._finish_fit(t0)
        logger.debug(
            f"TrainingDomain fit | {len(loo)} columns | "
            f"duration={self._fit_duration_seconds:.3f}s"
        )
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, query: pd.DataFrame) -> TrainingDomainRunResult:
        """Score query molecules against every training column.

        ``query`` needs an ``'input'`` SMILES column; ``key`` (if present)
        labels the rows of the details table.
        """
        self._check_fitted()
        assert self._training is not None and self._loo is not None
        if "input" not in query.columns:
            raise ValueError("TrainingDomain.run requires an 'input' SMILES column.")
        idx = list(query.index)
        keys = (
            query["key"].astype(str).tolist()
            if "key" in query.columns
            else [str(i) for i in idx]
        )
        std = [_standardize(s) for s in query["input"]]
        valid = np.array([s is not None for s in std])
        query_smiles = [s if s is not None else "" for s in std]

        names = self._training.column_names
        calibrated = np.full((len(query), len(names)), np.nan)
        raw = np.full((len(query), len(names)), np.nan)
        in_train = np.zeros((len(query), len(names)), dtype=bool)
        rows: list[dict[str, Any]] = []
        for j, name in enumerate(names):
            column = self._training.columns[name]
            vi = self._training.indices[name]
            k = min(N_NEIGHBORS, column.n - 1)
            sims, nn_idx, self_hit = _nearest_training(
                vi, [query_smiles[i] for i in np.flatnonzero(valid)], k
            )
            rows_valid = np.flatnonzero(valid)
            raw[rows_valid, j] = sims[:, 0]
            in_train[rows_valid, j] = self_hit
            calibrated[rows_valid, j] = _cdf_score(
                sims[:, 0], self._loo[name], higher_is_higher=True
            )
            for r, i in enumerate(rows_valid):
                rows.append(
                    {
                        "key": keys[i],
                        "column": name,
                        "domain": calibrated[i, j],
                        "domain_raw": raw[i, j],
                        "n_train": column.n,
                        "in_training": bool(self_hit[r]),
                        "nn_keys": "|".join(column.ids[t] for t in nn_idx[r]),
                        "nn_similarities": "|".join(f"{v:.3f}" for v in sims[r]),
                        "nn_y": (
                            "|".join(_fmt(column.y[t]) for t in nn_idx[r])
                            if column.y is not None
                            else ""
                        ),
                    }
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
            summary = np.nanquantile(calibrated, SUMMARY_QUANTILE, axis=1)
            summary_raw = np.nanquantile(raw, SUMMARY_QUANTILE, axis=1)
        details = pd.DataFrame(
            rows,
            columns=[
                "key",
                "column",
                "domain",
                "domain_raw",
                "n_train",
                "in_training",
                "nn_keys",
                "nn_similarities",
                "nn_y",
            ],
        )
        return TrainingDomainRunResult(
            score=pd.Series(summary, index=idx, name="training_domain"),
            score_raw=pd.Series(summary_raw, index=idx, name="training_domain_raw"),
            n_columns=pd.Series(
                np.isfinite(calibrated).sum(axis=1),
                index=idx,
                name="training_n_columns",
            ),
            in_training_any=pd.Series(
                in_train.any(axis=1), index=idx, name="in_training_any"
            ),
            details=details,
            metadata={
                "n_columns": len(names),
                "columns": names,
                "summary_quantile": SUMMARY_QUANTILE,
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._loo is not None and self._training is not None
        names = self._training.column_names
        np.savez(
            folder / SIMILARITIES_FILE,
            **{f"c{i:03d}": self._loo[n] for i, n in enumerate(names)},
        )
        with open(folder / STATE_FILE, "w") as f:
            json.dump(
                {"columns": names, "reference_domain": self._reference_domain},
                f,
                indent=2,
            )

    def _load_own(self, folder: pathlib.Path) -> None:
        payload = read_json(folder / STATE_FILE, self.NAME)
        names = payload["columns"]
        assert self._training is not None
        if names != self._training.column_names:
            raise ValueError(
                "training_domain/state.json columns do not match training/metadata.json."
            )
        with np.load(require_file(folder / SIMILARITIES_FILE, self.NAME)) as npz:
            self._loo = {n: np.asarray(npz[f"c{i:03d}"]) for i, n in enumerate(names)}
        self._reference_domain = dict(payload["reference_domain"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return self._training is not None and self._loo is not None

    @property
    def training_(self) -> TrainingFitState:
        self._check_fitted()
        assert self._training is not None
        return self._training

    @property
    def reference_domain_(self) -> dict[str, float]:
        """Mean calibrated domain of each column's own training molecules (≈ 0.5)."""
        self._check_fitted()
        assert self._reference_domain is not None
        return dict(self._reference_domain)


def _nearest_training(vi, query_smiles: list[str], k: int):
    """Top-k training neighbours per query, excluding the query's own entry.

    Returns ``(similarities (n, k) descending, indices (n, k), self_hit (n,))``.
    A query whose standardised SMILES equals a training SMILES drops that
    neighbour (leave-one-out, like the calibration table); others drop their
    (k+1)-th neighbour.
    """
    n = len(query_smiles)
    if n == 0:
        return (
            np.zeros((0, k)),
            np.zeros((0, k), dtype=np.int64),
            np.zeros(0, dtype=bool),
        )
    dist, nn = vi.query(query_smiles, k=k + 1, show_progress=False)
    train_smiles = vi.smiles
    drop = np.full(n, k, dtype=np.int64)
    for i, smi in enumerate(query_smiles):
        for j in range(k + 1):
            if train_smiles[nn[i, j]] == smi:
                drop[i] = j
                break
    keep = np.arange(k + 1)[None, :] != drop[:, None]
    sims = (1.0 - dist[keep].reshape(n, k)).astype(np.float64)
    return sims, nn[keep].reshape(n, k), drop != k


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"
