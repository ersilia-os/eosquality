"""XGBoost learner and SHAP-attribution helpers behind the Signal score.

:class:`SignalLearner` fits one XGBoost regressor from a chemical
descriptor matrix to the scaled model outputs; :func:`_shap_attributions`
and :func:`_signal_raw_from_attributions` turn its native TreeSHAP values
into the per-row Gini coefficient that :class:`~eosquality.scores.signal.Signal`
calibrates.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

from eosquality.scores._base import read_json, require_file
from eosquality.utils.logging import logger

LEARNER_STATE_FILE = "learner.json"


LEARNER_MODEL_FILE = "learner.ubj"


# Cap on the size of the eval_set XGBoost evaluates against every round
# for early-stopping. The canonical val slice is ~10% of a 1.35M-row
# reference (~135k); evaluating that many rows ~300 times per fit makes
# the XGBoost call O(minutes) on a ~1k-row training set even though the
# tree-building work itself is tiny. The early-stopping signal is set
# by the val curve's curvature, which is essentially identical at 5k
# rows. We still report ``r2_val`` on the FULL val slice (separately,
# after fit) and calibration ``sorted_self_aggregates`` is also built
# from the full val slice — only the per-round eval_set is sampled.
EARLY_STOP_VAL_MAX: int = 5000


EARLY_STOP_VAL_SEED: int = 0  # deterministic subsample for reproducibility


def _shap_attributions(
    model: xgb.XGBRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """Per-row ``|SHAP|`` attribution matrix, shape ``(n_samples, n_features)``.

    Uses XGBoost's native TreeSHAP (``Booster.predict(..., pred_contribs=True)``)
    rather than the ``shap`` library, which has a tree-dump parsing bug
    against XGBoost ≥3.0 (leaf values are list-wrapped). For multi-output
    models, ``|SHAP|`` is summed across outputs so the returned matrix is
    always 2-D — one row per input, one column per feature.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X)
    # pred_contribs returns SHAP values: per-feature contributions per row,
    # with one extra trailing column for the bias term.
    # - single output: shape (n, n_features + 1)
    # - multi-output:  shape (n, n_outputs, n_features + 1)   ← XGBoost layout
    contribs = booster.predict(dmat, pred_contribs=True)
    if contribs.ndim == 2:
        # drop bias column → (n, n_features), then absolute value
        return np.abs(contribs[:, :-1])
    if contribs.ndim == 3:
        # drop bias column on axis=2 (last) → (n, n_outputs, n_features);
        # aggregate across outputs by summing |SHAP|.
        return np.abs(contribs[:, :, :-1]).sum(axis=1)
    raise RuntimeError(
        f"Unexpected pred_contribs shape {contribs.shape}; "
        "expected 2-D or 3-D from Booster.predict(pred_contribs=True)."
    )


def _signal_raw_from_attributions(attribution: np.ndarray) -> np.ndarray:
    """Reduce a SHAP attribution matrix to one Gini score per row.

    Pure-arithmetic counterpart to :func:`_shap_signal_raw` that operates
    on an already-computed ``(n_samples, n_features)`` ``|SHAP|`` matrix.

    Returns the Gini coefficient of each row's ``|SHAP|`` distribution
    in ``[0, 1]``:

    - **Gini ≈ 0** — attribution is spread roughly uniformly across
      features (the model isn't keying on a small set; "scattered"
      chemistry).
    - **Gini → 1** — one feature carries most of the attribution
      ("focused" chemistry).

    Computed via the Lorenz-curve identity on descending-sorted data:
    ``G = 2 · mean(cumulative_fraction) − 1``. Rows with degenerate
    (~zero) total attribution fall back to ``0`` by convention (a
    zero-mass model can't be "focused").
    """
    n_samples, n_features = attribution.shape
    total = attribution.sum(axis=1, keepdims=True)
    safe_total = np.maximum(total, 1e-12)
    # Sort each row descending so the Lorenz cumulative fraction starts
    # large (max-share feature first) and walks toward 1.
    sorted_attr = np.sort(attribution, axis=1)[:, ::-1]
    cum_frac = np.cumsum(sorted_attr, axis=1) / safe_total
    gini = 2.0 * cum_frac.mean(axis=1) - 1.0
    # Clip into [0, 1] (finite-n bias can push a uniform row slightly
    # below 0; one-feature-dominant rows hit exactly 1 - 1/n).
    gini = np.clip(gini, 0.0, 1.0)
    gini = np.where(total.squeeze(-1) > 1e-12, gini, 0.0)
    return gini


def _shap_signal_raw(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """End-to-end raw signal per row: SHAP → Gini.

    Convenience composition of :func:`_shap_attributions` and
    :func:`_signal_raw_from_attributions` for callers that don't need
    the raw attribution matrix (i.e., the run path). The calibration
    path calls the two underlying helpers separately so it can persist
    the full attribution matrix for offline experimentation without
    recomputing SHAP.
    """
    return _signal_raw_from_attributions(_shap_attributions(model, X))


class SignalLearner:
    """XGBoost regressor from descriptor matrix to scaled model outputs.

    Holds the fitted model, the best iteration discovered via early
    stopping on the validation slice, and the per-output validation R².
    Persisted by :class:`Signal` as ``signal/learner.ubj`` (model) and
    ``signal/learner.json`` (``best_iteration``, ``r2_val``, params).
    """

    def __init__(self) -> None:
        self._model: xgb.XGBRegressor | None = None
        self._best_iteration: int | None = None
        self._output_columns: list[str] | None = None
        self._r2_val: np.ndarray | None = None
        self._params: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit_from_arrays(
        self,
        *,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        output_columns: list[str],
        n_estimators_min: int = 100,
        n_estimators_max: int = 500,
        early_stopping_rounds: int = 25,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 0,
    ) -> SignalLearner:
        """Train the XGBoost regressor with early stopping on the val slice.

        If early stopping fires below ``n_estimators_min`` rounds, the model is
        retrained for exactly ``n_estimators_min`` rounds (a safety net against
        premature stops on small or noisy val slices).

        Parameters
        ----------
        X_train, Y_train, X_val, Y_val : numpy.ndarray
            Descriptor matrices and scaled outputs of the train and val rows,
            without NaN targets.
        output_columns : list of str
            Names of the columns of ``Y_train``.
        n_estimators_min, n_estimators_max : int, optional
            Bounds on the number of boosting rounds.
        early_stopping_rounds : int, optional
            Patience of early stopping on the val RMSE.
        learning_rate, max_depth, random_state : optional
            XGBoost hyperparameters.

        Returns
        -------
        SignalLearner
            ``self``, fitted (model, ``best_iteration_``, per-output ``r2_val_``).
        """
        t0 = time.perf_counter()
        # One tree per output (not "multi_output_tree"): vector-leaf trees do
        # not support pred_contribs (TreeSHAP), which Signal needs at run time.
        params = dict(
            tree_method="hist",
            n_estimators=n_estimators_max,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        model, best_iteration = _train(
            params,
            X_train,
            Y_train,
            _early_stopping_set(X_val, Y_val),
            n_estimators_min,
        )

        r2_val = np.atleast_1d(
            r2_score(Y_val, model.predict(X_val), multioutput="raw_values")
        )
        self._model = model
        self._best_iteration = best_iteration
        self._output_columns = list(output_columns)
        self._r2_val = r2_val.astype(np.float64)
        self._params = {
            "n_estimators_min": int(n_estimators_min),
            "n_estimators_max": int(n_estimators_max),
            "early_stopping_rounds": int(early_stopping_rounds),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "random_state": int(random_state),
        }
        logger.info(
            f"signal.learner | n_train={len(X_train):,} n_val={len(X_val):,} | "
            f"best_iteration={best_iteration} r2_val mean={float(np.mean(r2_val)):.4f} "
            f"| {time.perf_counter() - t0:.1f}s"
        )
        return self

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, folder: pathlib.Path) -> None:
        """Write ``learner.json`` + ``learner.ubj`` into ``folder``."""
        self._check_fitted()
        assert self._model is not None
        assert self._r2_val is not None
        payload = {
            "best_iteration": int(self._best_iteration),  # type: ignore[arg-type]
            "output_columns": list(self._output_columns or []),
            "r2_val": self._r2_val.tolist(),
            "params": dict(self._params or {}),
        }
        with open(folder / LEARNER_STATE_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        self._model.save_model(str(folder / LEARNER_MODEL_FILE))

    @classmethod
    def load(cls, folder: pathlib.Path) -> SignalLearner:
        """Reconstruct from ``learner.json`` + ``learner.ubj`` in ``folder``."""
        payload = read_json(folder / LEARNER_STATE_FILE, "signal")
        model = xgb.XGBRegressor()
        model.load_model(str(require_file(folder / LEARNER_MODEL_FILE, "signal")))
        instance = cls()
        instance._model = model
        instance._best_iteration = int(payload["best_iteration"])
        instance._output_columns = list(payload["output_columns"])
        instance._r2_val = np.asarray(payload["r2_val"], dtype=np.float64)
        instance._params = dict(payload.get("params", {}))
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._model is not None
            and self._best_iteration is not None
            and self._r2_val is not None
        )

    @property
    def model_(self) -> xgb.XGBRegressor:
        self._check_fitted()
        assert self._model is not None
        return self._model

    @property
    def best_iteration_(self) -> int:
        self._check_fitted()
        assert self._best_iteration is not None
        return self._best_iteration

    @property
    def output_columns_(self) -> list[str]:
        self._check_fitted()
        assert self._output_columns is not None
        return list(self._output_columns)

    @property
    def r2_val_(self) -> np.ndarray:
        self._check_fitted()
        assert self._r2_val is not None
        return self._r2_val

    @property
    def params_(self) -> dict[str, Any]:
        self._check_fitted()
        return dict(self._params or {})

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("SignalLearner must be fitted (or loaded) before use.")


def _train(params, X_train, Y_train, eval_pair, n_estimators_min):
    """Fit with early stopping; refit for ``n_estimators_min`` rounds if it stops early."""
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, Y_train, eval_set=[eval_pair], verbose=False)
    best_iteration = int(model.best_iteration)
    if best_iteration + 1 >= n_estimators_min:
        return model, best_iteration
    logger.info(
        f"signal.learner | early stop at {best_iteration + 1} < "
        f"n_estimators_min={n_estimators_min}; retraining without early stopping"
    )
    params = {**params, "n_estimators": n_estimators_min}
    params.pop("early_stopping_rounds")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, Y_train, eval_set=[eval_pair], verbose=False)
    # XGBoost 3.x has no .best_iteration without early stopping.
    return model, int(n_estimators_min) - 1


def _early_stopping_set(
    X_val: np.ndarray, Y_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Val rows XGBoost evaluates every boosting round, capped at EARLY_STOP_VAL_MAX.

    The full val slice is still used for ``r2_val`` and for calibration.
    """
    if len(X_val) <= EARLY_STOP_VAL_MAX:
        return X_val, Y_val
    rng = np.random.default_rng(EARLY_STOP_VAL_SEED)
    idx = np.sort(rng.choice(len(X_val), size=EARLY_STOP_VAL_MAX, replace=False))
    return X_val[idx], Y_val[idx]
