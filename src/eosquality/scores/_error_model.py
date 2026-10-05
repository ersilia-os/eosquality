"""Per-endpoint error model behind :class:`TrainingDifficulty`.

Follows the error models of UNIQUE (Novartis; ``unique.pipeline`` and
``unique.error_models``, adapted from DEUP, Lahlou et al. 2021), applied to
a surrogate because the Ersilia model is a black box. For one output column
with labels:

1. A **surrogate** random forest on Morgan bits is fitted with scaffold-grouped
   CV. Every training molecule gets an out-of-fold (OOF) prediction ``ŷ``
   (P(y = 1) for binary labels) and tree variance; labelled ones get the OOF
   residual ``r = |y − ŷ|`` (UNIQUE's L1 error).
2. Inputs are grouped as in UNIQUE:

   - **base UQ**: mean Tanimoto distance to the ``k`` nearest *other*
     training molecules, three KDE log-densities
     (:mod:`eosquality.scores._density`), the surrogate's ensemble variance,
     and, for binary labels, the top-1 class probability ``max(p, 1 − p)``;
   - **transformed UQ**: DiffkNN on the prediction and on the variance,
     ``|v_i − mean(v over the k nearest training molecules)|``, plus two
     eosquality additions that use labels (not in UNIQUE): the
     similarity-weighted mean OOF residual of the neighbours
     (``local_oof_error``) and the spread of their labels (``label_spread``);
   - **data features**: the 166 MACCS keys;
   - **prediction**: ``ŷ``.

3. As in UNIQUE, an **error model** random forest is fitted on three feature
   sets (``VARIANTS``): data + base + prediction, base + prediction, and
   transformed + prediction. Each gets OOF predictions on the same folds;
   the one with the highest Spearman correlation to ``r`` is kept, and its
   OOF predictions are the calibration table. (UNIQUE picks its best method
   on a held-out test split by bootstrap and Wilcoxon tests instead.)

Training molecules are left out of their own neighbours and KDE kernel
(UNIQUE keeps them). Models are persisted with joblib (pickle): only load
artifacts from a trusted source. The scikit-learn version is recorded and
checked on load.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from dataclasses import dataclass, field
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
from eosquality.scores._training_helpers import _nearest_training
from eosquality.training.data import TrainingColumn
from eosquality.training.folds import cv_folds, morgan_bits
from eosquality.vectorindex import VectorIndex

# Minimum labelled training molecules for an endpoint to get an error model.
MIN_LABELLED = 50
N_FOLDS = 5
SEED = 0
# UNIQUE's three error-model feature sets, by block.
VARIANTS = {
    "data+base+pred": ("data", "base", "pred"),
    "base+pred": ("base", "pred"),
    "transformed+pred": ("transformed", "pred"),
}
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
    variant: str  # chosen feature set (a ``VARIANTS`` key)
    surrogate: Any
    density: TrainingDensity
    error_model: Any
    residuals: np.ndarray  # (n,) OOF |y − ŷ| per training molecule (NaN: no label)
    oof_prediction: np.ndarray  # (n,) OOF surrogate prediction (DiffkNN neighbours)
    oof_variance: np.ndarray  # (n,) OOF surrogate tree variance (DiffkNN neighbours)
    sorted_oof_error: np.ndarray  # ascending OOF predicted errors (CDF table)
    n_labelled: int
    variant_spearman: dict[str, float] = field(default_factory=dict)
    cv: str = "scaffold"  # "scaffold", or "random" when scaffolds cannot split
    oof_error: np.ndarray | None = None  # (n,) OOF predicted error per molecule

    @property
    def spearman(self) -> float:
        """Spearman(OOF predicted error, OOF residual) of the chosen variant.

        Returns
        -------
        float
        """
        return self.variant_spearman[self.variant]

    @property
    def features(self) -> list[str]:
        """Input names of the chosen variant, in column order.

        Returns
        -------
        list of str
        """
        return _variant_names(self.variant, self.binary)

    def predict(
        self, column: TrainingColumn, vi: VectorIndex, smiles: list[str]
    ) -> np.ndarray:
        """Predicted |error| for standardised query SMILES.

        Parameters
        ----------
        column : TrainingColumn
            This endpoint's training set (labels for the neighbour inputs).
        vi : VectorIndex
            This endpoint's Morgan index.
        smiles : list of str
            Standardised, valid query SMILES.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` predicted absolute error, in the endpoint's label units.
        """
        if not smiles:
            return np.zeros(0)
        dist, nn, _ = _nearest_training(vi, smiles, self.k)
        surrogate = _surrogate_predict(self.surrogate, morgan_bits(smiles), self.binary)
        # A query that is a training molecule gets exactly its fit-time inputs:
        # its out-of-fold prediction and variance (the final surrogate saw its
        # label) and its self-kNN neighbours (FPSim2 may order ties differently).
        index = {s: i for i, s in enumerate(column.smiles)}
        rows = np.array([index.get(s, -1) for s in smiles])
        known = rows >= 0
        if known.any():
            t = rows[known]
            dist[known] = vi.self_knn_distances(self.k)[t]
            nn[known] = vi.self_knn_indices(self.k)[t]
            surrogate[known, 0] = self.oof_prediction[t]
            surrogate[known, 1] = self.oof_variance[t]
        maccs = compute_maccs(smiles, show_progress=False)
        position = {
            column.smiles[r]: p for p, r in enumerate(self.density.reference_rows)
        }
        self_positions = np.array([position.get(s, -1) for s in smiles])
        blocks = _blocks(
            dist=dist,
            nn=nn,
            y=column.y,
            residuals=self.residuals,
            prediction=surrogate[:, 0],
            variance=surrogate[:, 1],
            neighbour_prediction=self.oof_prediction,
            neighbour_variance=self.oof_variance,
            log_density=self.density.log_density(maccs, self_positions),
            maccs=maccs,
            binary=self.binary,
        )
        predicted = self.error_model.predict(_variant_matrix(blocks, self.variant))
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
            "variant": self.variant,
            "variant_spearman": self.variant_spearman,
            "cv": self.cv,
            "n_labelled": self.n_labelled,
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
            variant=state["variant"],
            surrogate=joblib.load(require_file(folder / SURROGATE_FILE, component)),
            density=joblib.load(require_file(folder / DENSITY_FILE, component)),
            error_model=joblib.load(require_file(folder / ERROR_MODEL_FILE, component)),
            residuals=arrays["residuals"],
            oof_prediction=arrays["oof_prediction"],
            oof_variance=arrays["oof_variance"],
            sorted_oof_error=arrays["sorted_oof_error"],
            oof_error=arrays.get("oof_error"),
            n_labelled=int(state["n_labelled"]),
            variant_spearman={
                k: float(v) for k, v in state["variant_spearman"].items()
            },
            cv=state.get("cv", "scaffold"),
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
    """Fit the surrogate, the densities and the best error-model variant.

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
    y = column.y
    binary = column.y_kind == "binary"
    labelled = np.isfinite(y)
    X = morgan_bits(column.smiles)
    folds, cv = cv_folds(column.smiles, labelled, N_FOLDS, SEED)
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
    maccs = compute_maccs(column.smiles, show_progress=False)
    density = TrainingDensity.fit(maccs, SEED)
    self_positions = np.full(column.n, -1)
    self_positions[density.reference_rows] = np.arange(len(density.reference_rows))
    blocks = _blocks(
        dist=vi.self_knn_distances(k),
        nn=vi.self_knn_indices(k),
        y=y,
        residuals=residuals,
        prediction=prediction,
        variance=variance,
        neighbour_prediction=prediction,
        neighbour_variance=variance,
        log_density=density.log_density(maccs, self_positions),
        maccs=maccs,
        binary=binary,
    )
    variant, variant_spearman, oof_error = _select_variant(blocks, residuals, folds)
    target = np.isfinite(residuals)
    error_model = _new_error_model().fit(
        _variant_matrix(blocks, variant)[target], residuals[target]
    )
    return EndpointErrorModel(
        name=column.name,
        binary=binary,
        k=k,
        variant=variant,
        surrogate=_new_surrogate(binary).fit(X[labelled], _target(y[labelled], binary)),
        density=density,
        error_model=error_model,
        residuals=residuals,
        oof_prediction=prediction,
        oof_variance=variance,
        sorted_oof_error=np.sort(oof_error[target & np.isfinite(oof_error)]),
        oof_error=oof_error,
        n_labelled=int(labelled.sum()),
        variant_spearman=variant_spearman,
        cv=cv,
    )


def _select_variant(blocks, residuals, folds):
    """OOF-fit every variant; return ``(best, spearman per variant, best OOF)``."""
    target = np.isfinite(residuals)
    spearman, oof = {}, {}
    for variant in VARIANTS:
        oof[variant] = _oof(
            _new_error_model,
            _variant_matrix(blocks, variant),
            residuals,
            target,
            folds,
            lambda m, x: m.predict(x)[:, None],
        )[:, 0]
        scored = target & np.isfinite(oof[variant])
        rho = spearmanr(oof[variant][scored], residuals[scored]).statistic
        spearman[variant] = float(rho) if np.isfinite(rho) else float("nan")
    best = max(
        VARIANTS,
        key=lambda v: spearman[v] if np.isfinite(spearman[v]) else -np.inf,
    )
    return best, spearman, oof[best]


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


def _base_names(binary: bool) -> tuple[str, ...]:
    top1 = ("probability_top1",) if binary else ()
    return ("knn_distance", *KDE_NAMES, "ensemble_variance", *top1)


TRANSFORMED_NAMES = (
    "diffknn_prediction",
    "diffknn_variance",
    "local_oof_error",
    "label_spread",
)


def _variant_names(variant: str, binary: bool) -> list[str]:
    names = {
        "data": MACCS_NAMES,
        "base": _base_names(binary),
        "transformed": TRANSFORMED_NAMES,
        "pred": ("prediction",),
    }
    return [n for block in VARIANTS[variant] for n in names[block]]


def _blocks(
    *,
    dist,
    nn,
    y,
    residuals,
    prediction,
    variance,
    neighbour_prediction,
    neighbour_variance,
    log_density,
    maccs,
    binary,
) -> dict[str, np.ndarray]:
    """Input blocks ``data``, ``base``, ``transformed`` and ``pred`` (rows aligned)."""
    sim = 1.0 - dist
    nn_res = residuals[nn]
    weights = np.where(np.isfinite(nn_res), sim, 0.0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        local_error = np.nansum(weights * np.nan_to_num(nn_res), axis=1) / (
            weights.sum(axis=1)
        )
        spread = np.nanstd(y[nn], axis=1)
        diff_pred = np.abs(prediction - np.nanmean(neighbour_prediction[nn], axis=1))
        diff_var = np.abs(variance - np.nanmean(neighbour_variance[nn], axis=1))
    base = [dist.mean(axis=1)[:, None], log_density, variance[:, None]]
    if binary:
        base.append(np.maximum(prediction, 1.0 - prediction)[:, None])
    return {
        "data": maccs.astype(np.float64),
        "base": np.hstack(base),
        "transformed": np.column_stack([diff_pred, diff_var, local_error, spread]),
        "pred": prediction[:, None],
    }


def _variant_matrix(blocks: dict[str, np.ndarray], variant: str) -> np.ndarray:
    return np.hstack([blocks[b] for b in VARIANTS[variant]]).astype(np.float64)
