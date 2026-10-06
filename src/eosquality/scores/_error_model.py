"""Per-endpoint error model behind :class:`TrainingDifficulty`.

Follows the error models of UNIQUE (Novartis; ``unique.pipeline`` and
``unique.error_models``, adapted from DEUP, Lahlou et al. 2021), applied to
a surrogate because the Ersilia model is a black box. For one output column
with labels:

1. A **surrogate** random forest on Morgan bits is fitted with scaffold-grouped
   CV (:func:`~eosquality.training.folds.cv_folds`). Every training molecule
   gets an out-of-fold (OOF) prediction ``ŷ`` (P(y = 1) for binary labels)
   and tree variance; labelled ones get the OOF residual ``r = |y − ŷ|``
   (UNIQUE's L1 error).
2. The inputs are UNIQUE's feature set (i), "data features + base UQ
   metrics + prediction" (``FEATURES``):

   - **data features**: the 166 MACCS keys;
   - **base UQ**: mean Tanimoto distance to the ``k`` nearest *other*
     training molecules, three KDE log-densities
     (:mod:`eosquality.scores._density`), the surrogate's ensemble variance,
     and, for binary labels, the top-1 class probability ``max(p, 1 − p)``;
   - **prediction**: ``ŷ``.

3. An **error model** random forest learns ``inputs → r``. Its own OOF
   predictions on the same folds are the calibration table, and their
   Spearman correlation with ``r`` is the honesty check (0 = no better than
   random).

Why only feature set (i): UNIQUE also builds error models on "base UQ +
prediction" and on "transformed UQ (DiffkNN) + prediction", and both UNIQUE
papers found set (i) best. On six MoleculeNet endpoints with scaffold
splits, scored against the held-out errors of three different models (see
``scripts/evaluate_training.py``), set (i) with MACCS had the best mean
Spearman (0.36, against 0.30 and 0.28 for the other two sets and 0.16 for
plain distance). Choosing the set per column by OOF Spearman picked the
transformed set most of the time and did worse: its neighbour-based inputs
(DiffkNN, neighbours' errors and label spread) are optimistic in OOF because
a training molecule's neighbours usually share its scaffold, hence its fold
and its fold-model's errors, which a new-scaffold query does not.

Training molecules are left out of their own neighbours and KDE kernel
(UNIQUE keeps them). Models are persisted with joblib (pickle): only load
artifacts from a trusted source. The scikit-learn version is recorded and
checked on load.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eosquality.exceptions import ArtifactVersionError
from eosquality.library.maccs import N_MACCS, compute_maccs
from eosquality.scores._base import read_json, require_file
from eosquality.scores._density import KDE_NAMES, TrainingDensity
from eosquality.scores._training_helpers import TrainingQuery
from eosquality.training.data import TrainingColumn
from eosquality.training.folds import cv_folds, morgan_bits
from eosquality.vectorindex import VectorIndex

# Minimum labelled training molecules for an endpoint to get an error model.
MIN_LABELLED = 50
# Most labelled molecules an endpoint's surrogate and error model are fitted
# on (a seeded random subset beyond it). Bounds fit time, artifact size and
# prediction time on large screens; training distance still uses every
# molecule. The held-out validation used the same cap.
MAX_FIT_MOLECULES = 10_000
N_FOLDS = 5
SEED = 0
MACCS_NAMES = tuple(f"maccs_{i}" for i in range(1, N_MACCS + 1))
SURROGATE_FILE = "surrogate.joblib"
DENSITY_FILE = "density.joblib"
ERROR_MODEL_FILE = "error_model.joblib"
ARRAYS_FILE = "arrays.npz"
STATE_FILE = "state.json"


@dataclass
class EndpointErrorModel:
    """Fitted surrogate, densities and error model for one output column."""

    name: str
    binary: bool
    k: int
    surrogate: Any
    density: TrainingDensity
    error_model: Any
    residuals: np.ndarray  # (n,) OOF |y − ŷ| per training molecule (NaN: no label)
    oof_prediction: np.ndarray  # (n,) OOF surrogate prediction (DiffkNN neighbours)
    oof_variance: np.ndarray  # (n,) OOF surrogate tree variance (DiffkNN neighbours)
    sorted_oof_error: np.ndarray  # ascending OOF predicted errors (CDF table)
    n_labelled: int
    spearman: float  # ρ(OOF predicted error, OOF residual): the honesty check
    cv: str = "scaffold"  # "scaffold", or "random" when scaffolds cannot split
    oof_error: np.ndarray | None = None  # (n,) OOF predicted error per molecule
    n_fit: int | None = None  # labelled molecules fitted on (MAX_FIT_MOLECULES cap)

    @property
    def features(self) -> list[str]:
        """Input names of the error model, in column order.

        Returns
        -------
        list of str
        """
        return feature_names(self.binary)

    def predict(
        self,
        column: TrainingColumn,
        vi: VectorIndex,
        features: TrainingQuery | list[str],
    ) -> np.ndarray:
        """Predicted |error| for standardised query SMILES.

        Parameters
        ----------
        column : TrainingColumn
            This endpoint's training set (labels for the neighbour inputs).
        vi : VectorIndex
            This endpoint's Morgan index.
        features : TrainingQuery or list of str
            The query's features (or its standardised, valid SMILES).

        Returns
        -------
        numpy.ndarray
            ``(n,)`` predicted absolute error, in the endpoint's label units.
        """
        if not isinstance(features, TrainingQuery):
            features = TrainingQuery(features)
        smiles = features.smiles
        if not smiles:
            return np.zeros(0)
        dist, _, _ = features.nearest(vi, self.k)
        dist = dist.copy()  # rows of training molecules are replaced below
        surrogate = _surrogate_predict(self.surrogate, features.morgan, self.binary)
        # A query that is a training molecule gets exactly its fit-time inputs:
        # its out-of-fold prediction and variance (the final surrogate saw its
        # label) and its self-kNN distances.
        index = {s: i for i, s in enumerate(column.smiles)}
        rows = np.array([index.get(s, -1) for s in smiles])
        # Training molecules the models were fitted on (others are new to them).
        known = rows >= 0
        known[known] = np.isfinite(self.oof_prediction[rows[known]])
        if known.any():
            t = rows[known]
            dist[known] = vi.self_knn_distances(self.k)[t]
            surrogate[known, 0] = self.oof_prediction[t]
            surrogate[known, 1] = self.oof_variance[t]
        maccs = features.maccs
        position = {
            column.smiles[r]: p for p, r in enumerate(self.density.reference_rows)
        }
        self_positions = np.array([position.get(s, -1) for s in smiles])
        inputs = _inputs(
            dist=dist,
            prediction=surrogate[:, 0],
            variance=surrogate[:, 1],
            log_density=self.density.log_density(maccs, self_positions),
            maccs=maccs,
            binary=self.binary,
        )
        predicted = self.error_model.predict(inputs)
        if known.any() and self.oof_error is not None:
            # The final error model saw training molecules; their out-of-fold
            # prediction is the value the calibration table was built from.
            own = self.oof_error[rows[known]]
            predicted[known] = np.where(np.isfinite(own), own, predicted[known])
        return predicted

    def save(self, folder: pathlib.Path) -> None:
        """Write the models, arrays and ``state.json`` into ``folder``.

        Parameters
        ----------
        folder : pathlib.Path
            Destination folder (created if missing).
        """
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.surrogate, folder / SURROGATE_FILE)
        joblib.dump(self.density, folder / DENSITY_FILE)
        joblib.dump(self.error_model, folder / ERROR_MODEL_FILE)
        np.savez(
            folder / ARRAYS_FILE,
            residuals=self.residuals,
            oof_prediction=self.oof_prediction,
            oof_variance=self.oof_variance,
            sorted_oof_error=self.sorted_oof_error,
            oof_error=self.oof_error,
        )
        state = {
            "name": self.name,
            "binary": self.binary,
            "k": self.k,
            "spearman": self.spearman,
            "cv": self.cv,
            "n_labelled": self.n_labelled,
            "n_fit": self.n_fit,
            "features": self.features,
            "sklearn_version": sklearn.__version__,
        }
        with open(folder / STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, folder: pathlib.Path, component: str) -> EndpointErrorModel:
        """Read a model written by :meth:`save`.

        Parameters
        ----------
        folder : pathlib.Path
            Folder written by :meth:`save`.
        component : str
            Component name for error messages.

        Returns
        -------
        EndpointErrorModel

        Raises
        ------
        ArtifactVersionError
            If the models were saved with another scikit-learn version.
        """
        state = read_json(folder / STATE_FILE, component)
        if state["sklearn_version"] != sklearn.__version__:
            raise ArtifactVersionError(
                f"{folder} was saved with scikit-learn {state['sklearn_version']}; "
                f"this install has {sklearn.__version__}. Refit the training modality."
            )
        with np.load(require_file(folder / ARRAYS_FILE, component)) as arrays:
            arrays = {key: arrays[key] for key in arrays.files}
        return cls(
            name=state["name"],
            binary=bool(state["binary"]),
            k=int(state["k"]),
            surrogate=joblib.load(require_file(folder / SURROGATE_FILE, component)),
            density=joblib.load(require_file(folder / DENSITY_FILE, component)),
            error_model=joblib.load(require_file(folder / ERROR_MODEL_FILE, component)),
            residuals=arrays["residuals"],
            oof_prediction=arrays["oof_prediction"],
            oof_variance=arrays["oof_variance"],
            sorted_oof_error=arrays["sorted_oof_error"],
            oof_error=arrays.get("oof_error"),
            n_labelled=int(state["n_labelled"]),
            spearman=float(state["spearman"]),
            cv=state.get("cv", "scaffold"),
            n_fit=state.get("n_fit"),
        )


def is_eligible(column: TrainingColumn) -> bool:
    """Whether ``column`` has enough labels for an error model.

    Parameters
    ----------
    column : TrainingColumn
        A training set.

    Returns
    -------
    bool
    """
    return column.y is not None and int(np.isfinite(column.y).sum()) >= MIN_LABELLED


def fit_endpoint(column: TrainingColumn, vi: VectorIndex, k: int) -> EndpointErrorModel:
    """Fit the surrogate, the densities and the error model for one column.

    Parameters
    ----------
    column : TrainingColumn
        Training set with labels (see :func:`is_eligible`).
    vi : VectorIndex
        The column's Morgan index (its self-kNN gives the neighbour inputs).
    k : int
        Neighbours per molecule.

    Returns
    -------
    EndpointErrorModel
    """
    fit_rows = _fit_rows(column.y)
    smiles = [column.smiles[i] for i in fit_rows]
    y = column.y[fit_rows]
    binary = column.y_kind == "binary"
    labelled = np.ones(len(fit_rows), dtype=bool)
    X = morgan_bits(smiles)
    folds, cv = cv_folds(smiles, labelled, N_FOLDS, SEED)
    surrogate_oof = _oof(
        lambda: _new_surrogate(binary),
        X,
        y,
        labelled,
        folds,
        lambda m, x: _surrogate_predict(m, x, binary),
        binary=binary,
    )
    prediction, variance = surrogate_oof[:, 0], surrogate_oof[:, 1]
    residuals = np.abs(y - prediction)
    maccs = compute_maccs(smiles, show_progress=False)
    density = TrainingDensity.fit(maccs, SEED)
    self_positions = np.full(len(smiles), -1)
    self_positions[density.reference_rows] = np.arange(len(density.reference_rows))
    density.reference_rows = fit_rows[density.reference_rows]  # column positions
    inputs = _inputs(
        dist=vi.self_knn_distances(k)[fit_rows],
        prediction=prediction,
        variance=variance,
        log_density=density.log_density(maccs, self_positions),
        maccs=maccs,
        binary=binary,
    )
    target = np.isfinite(residuals)
    oof_error = _oof(
        _new_error_model,
        inputs,
        residuals,
        target,
        folds,
        lambda m, x: m.predict(x)[:, None],
    )[:, 0]
    scored = target & np.isfinite(oof_error)
    rho = spearmanr(oof_error[scored], residuals[scored]).statistic

    def full(values: np.ndarray) -> np.ndarray:
        """Per-molecule values on the whole column (NaN outside the fit rows)."""
        out = np.full(column.n, np.nan)
        out[fit_rows] = values
        return out

    return EndpointErrorModel(
        name=column.name,
        binary=binary,
        k=k,
        surrogate=_new_surrogate(binary).fit(X[labelled], _target(y[labelled], binary)),
        density=density,
        error_model=_new_error_model().fit(inputs[target], residuals[target]),
        residuals=full(residuals),
        oof_prediction=full(prediction),
        oof_variance=full(variance),
        sorted_oof_error=np.sort(oof_error[scored]),
        n_labelled=int(np.isfinite(column.y).sum()),
        spearman=float(rho) if np.isfinite(rho) else float("nan"),
        cv=cv,
        oof_error=full(oof_error),
        n_fit=len(fit_rows),
    )


def _fit_rows(y: np.ndarray) -> np.ndarray:
    """Positions of the labelled molecules to fit on (``MAX_FIT_MOLECULES`` at most).

    A seeded random subset when there are more, kept in column order.
    """
    rows = np.flatnonzero(np.isfinite(y))
    if len(rows) > MAX_FIT_MOLECULES:
        rng = np.random.default_rng(SEED)
        rows = np.sort(rng.choice(rows, MAX_FIT_MOLECULES, replace=False))
    return rows


def _oof(make, X, y, mask, folds, predict, *, binary: bool = False) -> np.ndarray:
    """Out-of-fold ``(n, m)`` outputs of ``predict`` for every row.

    Each fold's model is trained on the ``mask`` rows of the other folds and
    predicts every row of its fold (rows outside ``mask`` included).
    """
    out = None
    for f in np.unique(folds):
        train = mask & (folds != f)
        test = folds == f
        if not train.any():
            continue
        model = make().fit(X[train], _target(y[train], binary))
        values = predict(model, X[test])
        if out is None:
            out = np.full((len(y), values.shape[1]), np.nan)
        out[test] = values
    return out if out is not None else np.full((len(y), 1), np.nan)


def _target(y: np.ndarray, binary: bool) -> np.ndarray:
    return y.astype(np.int64) if binary else y


def _new_surrogate(binary: bool):
    cls = RandomForestClassifier if binary else RandomForestRegressor
    return cls(n_estimators=100, min_samples_leaf=3, n_jobs=-1, random_state=SEED)


def _new_error_model():
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED
    )


def _surrogate_predict(model, X: np.ndarray, binary: bool) -> np.ndarray:
    """``(n, 2)``: prediction and variance across the forest's trees.

    The prediction is the regression value, or P(y = 1) for a classifier
    (0 when the training fold held a single class).
    """
    if not binary:
        per_tree = np.stack([t.predict(X) for t in model.estimators_])
    else:
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros((len(X), 2))
        j = classes.index(1)
        per_tree = np.stack([t.predict_proba(X)[:, j] for t in model.estimators_])
    return np.column_stack([per_tree.mean(axis=0), per_tree.var(axis=0)])


def feature_names(binary: bool) -> list[str]:
    """Names of the error model's inputs, in column order (``FEATURES``).

    Parameters
    ----------
    binary : bool
        Binary labels (adds the top-1 class probability).

    Returns
    -------
    list of str
    """
    top1 = ["probability_top1"] if binary else []
    return [
        *MACCS_NAMES,
        "knn_distance",
        *KDE_NAMES,
        "ensemble_variance",
        *top1,
        "prediction",
    ]


def _inputs(*, dist, prediction, variance, log_density, maccs, binary) -> np.ndarray:
    """``(n, len(feature_names(binary)))``: data features, base UQ, prediction."""
    base = [dist.mean(axis=1)[:, None], log_density, variance[:, None]]
    if binary:
        base.append(np.maximum(prediction, 1.0 - prediction)[:, None])
    return np.hstack([maccs, *base, prediction[:, None]]).astype(np.float64)
