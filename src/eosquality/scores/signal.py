"""Signal score: per-query SHAP-attribution-Gini on a chemical descriptor.

Train one XGBoost regressor on the reference library and reduce
per-query ``|SHAP|`` attributions to a Gini coefficient: high when
attribution is focused on a few features ("focused" chemistry), low
when attribution scatters across many features ("scattered"
chemistry).

The feature backend is chosen at fit time via :class:`Signal`'s
``descriptor`` argument:

- ``"physchem"`` (default) — the RDKit physicochemical descriptor set
  (``Descriptors._descList``; ~200 descriptors, exact count depends on the
  RDKit version), precomputed at library build time alongside the FP index.
- ``"maccs"`` — RDKit MACCS structural keys, precomputed at library build
  time as ``maccs.npy``.

Both descriptors feed the same Gini aggregator. The choice is baked
into the saved artifact via the ``descriptor`` field in
``umbrella.json`` and recovered at load time — there is no run-time
override.

``signal_raw = Gini(|SHAP|_per_feature)`` bounded in ``[0, 1]``,
**high = focused** (one feature carries most of the attribution).
``signal`` is the CDF rank of ``signal_raw`` against the reference
val slice's own values.

Two classes:

- :class:`SignalLearner` — the single XGBoost regressor on the chosen
  descriptor matrix. Trains on (a capped subset of) the canonical train
  slice with early stopping on val.
- :class:`Signal` — the umbrella score component. Composes the
  descriptor backend + learner + SHAP-Gini aggregator + val-slice
  calibration. Persists per-backend state so the artifact is
  self-contained at run time, plus the full val-slice ``|SHAP|``
  matrix so the raw-score formula can be iterated offline.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from eosquality.scores._descriptors import (
    DEFAULT_DESCRIPTOR,
    DESCRIPTOR_NAMES,
    DescriptorBackend,
    load_backend,
    make_backend,
)
from eosquality.exceptions import ArtifactVersionError
from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import _reference_repr, _score_from_aggregates
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

SUBFOLDER = "signal"
LEARNER_STATE_FILE = "learner.json"
LEARNER_MODEL_FILE = "learner.ubj"
UMBRELLA_FILE = "umbrella.json"
SELF_AGGREGATES_FILE = "reference_self_aggregates.npy"
# Calibration-time SHAP matrix on the val slice — ``(n_val, n_features)``
# float32. Persisted so the score formula can be iterated offline without
# recomputing SHAP (which is the slow step). Provisional while the raw
# signal formula is in flux.
VAL_SHAP_ATTRIBUTIONS_FILE = "val_shap_attributions.npy"

# Version tag baked into the umbrella. Bumped to ``gini_v2`` when the
# umbrella schema gained the required ``descriptor`` field so legacy
# physchem-only artifacts fail load with a clear "refit" message instead
# of silently scoring against a backend that wasn't recorded.
SIGNAL_FORMULA_VERSION: str = "gini_v2"

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _normalize_y(
    reference: pd.DataFrame, shared: SharedFitState
) -> tuple[np.ndarray, list[str]]:
    """Scaled reference outputs projected onto the selected columns.

    Returns ``(Y, output_columns)``: the float32 ``(n_ref, n_selected)``
    eosframes-scaled matrix (reused from ``shared.ref_repr`` when aligned)
    and the selected column names in schema order, so Signal is trained on
    the same reduced output set the other scores see.
    """
    Y = _reference_repr(shared, reference).astype(np.float32)
    return Y, list(shared.selected_columns)


def _train_subset(
    train_indices: np.ndarray, max_train_samples: int | None
) -> np.ndarray:
    """Cap the training rows Signal fits on.

    ``train_indices`` is already a seeded random permutation slice, so the
    first ``max_train_samples`` entries are a random subset. ``None`` or
    non-positive values keep the full train slice. The shared split itself
    is never modified.
    """
    if max_train_samples is None or max_train_samples <= 0:
        return train_indices
    if max_train_samples >= len(train_indices):
        logger.debug(
            f"signal | train-subsample skipped (max_train_samples="
            f"{max_train_samples} ≥ n_train={len(train_indices):,})"
        )
        return train_indices
    logger.debug(
        f"signal | subsampling training set | n_train="
        f"{len(train_indices):,} → {max_train_samples:,}"
    )
    return train_indices[:max_train_samples]


# ---------------------------------------------------------------------------
# SignalLearner — real Y, early stopping on val
# ---------------------------------------------------------------------------


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
    ) -> "SignalLearner":
        """Train an XGBoost regressor from already-prepared X/Y arrays.

        Parameters
        ----------
        X_train, Y_train, X_val, Y_val:
            Pre-sliced descriptor matrix + normalized Y matrix for the
            canonical train + val rows. Caller (typically
            :meth:`Signal.fit`) is responsible for choosing the right
            descriptor backend, applying the eosframes scaler to Y, and
            dropping NaN rows.
        output_columns:
            Column names corresponding to ``Y_train.shape[1]`` — the
            schema-order list returned by
            :func:`_normalize_y`.
        n_estimators_min:
            Lower bound on boosting rounds. If early stopping fires
            below this many rounds, the model is retrained with
            ``n_estimators=n_estimators_min`` and no early stopping —
            a safety net against premature stops on small / noisy
            val slices.
        n_estimators_max:
            Upper bound on boosting rounds; early stopping will prune
            below this.
        early_stopping_rounds:
            Stop if val RMSE hasn't improved for this many rounds.
        learning_rate, max_depth, random_state:
            XGBoost hyperparameters.

        Records ``best_iteration_``, ``r2_val_`` (per-output array),
        ``output_columns_``, the trained model, and the training
        parameters.
        """
        t0 = time.perf_counter()

        logger.info(
            f"signal.learner | split | n_train={len(X_train):,}"
            f" n_val={len(X_val):,} n_outputs={Y_train.shape[1]}"
            f" n_features={X_train.shape[1]}"
        )

        params = dict(
            tree_method="hist",
            n_estimators=n_estimators_max,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        # NOTE: deliberately NOT setting ``multi_strategy="multi_output_tree"``
        # even when n_outputs > 1. XGBoost 3.x's vector-leaf trees don't yet
        # support ``predict(pred_contribs=True)`` (TreeSHAP), which is what
        # the Signal score uses at run time. The default (one-tree-per-output)
        # has slightly less efficient multi-output coupling but supports
        # native TreeSHAP and gives equivalent predictive quality here.

        # Cap the early-stopping eval_set so XGBoost doesn't predict on the
        # whole val slice every boosting round. The full val slice is still
        # used afterwards for the reported ``r2_val`` and (in the umbrella)
        # for calibration.
        if len(X_val) > EARLY_STOP_VAL_MAX:
            es_rng = np.random.default_rng(EARLY_STOP_VAL_SEED)
            es_idx = es_rng.choice(len(X_val), size=EARLY_STOP_VAL_MAX, replace=False)
            es_idx.sort()
            X_es, Y_es = X_val[es_idx], Y_val[es_idx]
            es_note = f"early-stop eval sampled to {EARLY_STOP_VAL_MAX:,} rows"
        else:
            X_es, Y_es = X_val, Y_val
            es_note = f"early-stop eval on full val slice ({len(X_val):,} rows)"

        t_train = time.perf_counter()
        logger.info(
            f"signal.learner | XGBoost fitting (real Y) | n_estimators_max="
            f"{n_estimators_max} early_stopping_rounds={early_stopping_rounds} "
            f"learning_rate={learning_rate} max_depth={max_depth} | {es_note}"
        )
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            Y_train,
            eval_set=[(X_es, Y_es)],
            verbose=False,
        )
        logger.info(
            f"signal.learner | XGBoost done | best_iteration={int(model.best_iteration)}"
            f" | {time.perf_counter() - t_train:.1f}s"
        )

        best_iteration = int(model.best_iteration)
        # If early stopping fired below the minimum, retrain a fixed-length
        # model with no early stopping. The val slice on tiny corpora can be
        # too noisy to trust the patience signal; this safety net keeps the
        # tree count honest regardless.
        if best_iteration + 1 < n_estimators_min:
            logger.info(
                f"signal.learner | best_iteration={best_iteration + 1} "
                f"< n_estimators_min={n_estimators_min}; retraining with "
                f"n_estimators={n_estimators_min} and no early stopping"
            )
            t_retrain = time.perf_counter()
            retrain_params = dict(params)
            retrain_params.pop("early_stopping_rounds", None)
            retrain_params["n_estimators"] = n_estimators_min
            model = xgb.XGBRegressor(**retrain_params)
            model.fit(X_train, Y_train, eval_set=[(X_es, Y_es)], verbose=False)
            # XGBoost 3.x raises AttributeError on .best_iteration when
            # early stopping isn't used, so derive it from n_estimators.
            best_iteration = int(n_estimators_min) - 1
            logger.info(
                f"signal.learner | retrain done | n_trees={n_estimators_min} | "
                f"{time.perf_counter() - t_retrain:.1f}s"
            )

        pred_val = model.predict(X_val)
        r2_val = np.atleast_1d(r2_score(Y_val, pred_val, multioutput="raw_values"))

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
            f"signal.learner | done | best_iteration={best_iteration}"
            f" r2_val mean={float(np.mean(r2_val)):.4f}"
            f" | total {time.perf_counter() - t0:.1f}s"
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
    def load(cls, folder: pathlib.Path) -> "SignalLearner":
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


# ---------------------------------------------------------------------------
# Signal — umbrella score: descriptor learner + SHAP-attribution Gini
# ---------------------------------------------------------------------------


@dataclass
class SignalRunResult:
    """Result returned by :meth:`Signal.run`."""

    score: pd.Series  # (n_query,) calibrated signal score in (0, 1]
    score_raw: pd.Series  # (n_query,) Gini(|SHAP| per feature) in [0, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


class Signal(ScoreComponent):
    """Per-query model-signal score via SHAP-attribution Gini.

    Fits a single XGBoost regressor (:class:`SignalLearner`) on the
    reference library, using the descriptor chosen at fit time (RDKit
    physchem descriptors by default, or MACCS keys). At run time, for each
    query, computes the per-query ``|SHAP|`` attribution over those
    features (summed across outputs when multi-output) and reduces it to
    the Gini coefficient: high when one (or a few) features carry most of
    the attribution, low when attribution is spread uniformly.

    Persists the XGBoost model, per-backend state (e.g. the physchem
    scaler params), the val-slice CDF lookup and the full val-slice
    ``|SHAP|`` matrix under ``<root>/signal/``. The descriptor identifier
    is recorded in ``umbrella.json`` so loaded artifacts run against the
    same backend they were trained on.
    """

    NAME = SUBFOLDER

    def __init__(self) -> None:
        super().__init__()
        self._learner: SignalLearner | None = None
        self._output_columns: list[str] | None = None
        self._backend: DescriptorBackend | None = None
        self._sorted_self_aggregates: np.ndarray | None = None
        self._reference_signal: float | None = None
        self._reference_signal_raw: float | None = None
        # (n_val, n_features) raw |SHAP| matrix on the calibration val
        # slice, persisted so the score formula can be iterated offline
        # without re-running SHAP.
        self._val_shap_attributions: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        *,
        vector_index: str | pathlib.Path | VectorIndex,
        shared: SharedFitState,
        descriptor: str = DEFAULT_DESCRIPTOR,
        max_train_samples: int | None = None,
        **learner_kwargs: Any,
    ) -> "Signal":
        """Fit the XGBoost regressor + calibrate via val-slice SHAP.

        Parameters
        ----------
        reference, vector_index, shared:
            The reference DataFrame, the vector index (path or instance)
            whose folder holds the library descriptor matrices, and the
            shared fit state.
        descriptor:
            Feature backend identifier: ``"physchem"`` (default) or
            ``"maccs"``. Persisted in ``umbrella.json`` and recovered at
            load time.
        max_train_samples:
            Optional cap on the number of training rows. ``None`` (or
            non-positive) → use the full ``shared.splits.train_indices``.
            The full val slice is always used for the reported ``r2_val``
            and for the calibration distribution; only the per-round
            early-stopping eval set is sampled to ``EARLY_STOP_VAL_MAX``.
        learner_kwargs:
            Forwarded to :meth:`SignalLearner.fit_from_arrays` (e.g.
            ``max_depth``, ``learning_rate``).
        """
        if descriptor not in DESCRIPTOR_NAMES:
            raise ValueError(
                f"Unknown signal descriptor {descriptor!r}; expected one of "
                f"{DESCRIPTOR_NAMES}."
            )
        t0 = time.perf_counter()
        vi = (
            vector_index
            if isinstance(vector_index, VectorIndex)
            else VectorIndex.load(pathlib.Path(vector_index))
        )
        backend = make_backend(descriptor, vi)
        n_features = backend.n_features

        Y, cols = _normalize_y(reference, shared)
        non_nan = ~np.isnan(Y).any(axis=1)
        train_idx = _train_subset(shared.splits.train_indices, max_train_samples)
        train_idx = train_idx[non_nan[train_idx]]
        val_idx = shared.splits.val_indices[non_nan[shared.splits.val_indices]]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                "Signal.fit needs at least one train and one val row whose "
                f"selected outputs are all finite (got n_train={len(train_idx)}, "
                f"n_val={len(val_idx)})."
            )
        logger.info(
            f"signal | fit | descriptor={descriptor} n_features={n_features} "
            f"n_train={len(train_idx):,} n_val={len(val_idx):,} "
            f"formula={SIGNAL_FORMULA_VERSION}"
        )

        X_train = backend.compute_reference_subset(reference, train_idx).astype(
            np.float32, copy=False
        )
        X_val = backend.compute_reference_subset(reference, val_idx).astype(
            np.float32, copy=False
        )
        learner = SignalLearner().fit_from_arrays(
            X_train=X_train,
            Y_train=Y[train_idx],
            X_val=X_val,
            Y_val=Y[val_idx],
            output_columns=cols,
            **learner_kwargs,
        )

        # Calibration on the val slice: the full (n_val, n_features) |SHAP|
        # matrix → Gini per row → sorted CDF anchor for queries at run time.
        # Orientation: high raw Gini = focused chemistry → high signal.
        t_cal = time.perf_counter()
        val_attribution = _shap_attributions(learner.model_, X_val)
        ref_agg = _signal_raw_from_attributions(val_attribution)
        sorted_self = np.sort(ref_agg).astype(np.float64)

        self._shared = shared
        self._learner = learner
        self._output_columns = list(shared.selected_columns)
        self._backend = backend
        self._sorted_self_aggregates = sorted_self
        self._reference_signal = float(
            np.mean(_score_from_aggregates(ref_agg, sorted_self))
        )
        self._reference_signal_raw = float(ref_agg.mean())
        # float32 halves disk + memory; SHAP values don't need float64.
        self._val_shap_attributions = val_attribution.astype(np.float32)
        self._finish_fit(t0)
        logger.info(
            f"signal | calibration done | reference_signal={self._reference_signal:.4f} "
            f"gini median={float(np.median(ref_agg)):.4f} "
            f"range=[{float(ref_agg.min()):.4f}, {float(ref_agg.max()):.4f}] | "
            f"{time.perf_counter() - t_cal:.1f}s (total {self._fit_duration_seconds:.1f}s)"
        )
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, query: pd.DataFrame) -> SignalRunResult:
        """Score query molecules from their SMILES.

        Computes the fitted descriptor for each query via the saved
        backend, runs the XGBoost regressor + native TreeSHAP, and reduces
        each row's ``|SHAP|`` attribution to a Gini coefficient calibrated
        against the reference val slice.

        Parameters
        ----------
        query:
            DataFrame with an ``'input'`` SMILES column.
        """
        self._check_fitted()
        assert self._learner is not None
        assert self._backend is not None
        assert self._sorted_self_aggregates is not None

        if "input" not in query.columns:
            raise ValueError(
                "Signal.run requires an 'input' column with SMILES strings."
            )
        t0 = time.perf_counter()
        smiles_list = list(query["input"])
        if smiles_list:
            query_X = self._backend.query_matrix(smiles_list).astype(
                np.float32, copy=False
            )
            row_aggregate = _shap_signal_raw(self._learner.model_, query_X)
        else:
            row_aggregate = np.zeros(0, dtype=np.float64)
        score = _score_from_aggregates(row_aggregate, self._sorted_self_aggregates)
        logger.debug(
            f"signal | run | {len(smiles_list):,} queries | "
            f"descriptor={self._backend.name} | {time.perf_counter() - t0:.1f}s"
        )
        idx = list(query.index)
        return SignalRunResult(
            score=pd.Series(score, index=idx, name="signal"),
            score_raw=pd.Series(row_aggregate, index=idx, name="signal_raw"),
            metadata={
                "descriptor": self._backend.name,
                "reference_signal": self._reference_signal,
                "reference_signal_raw": self._reference_signal_raw,
                "formula_version": SIGNAL_FORMULA_VERSION,
                "n_outputs": int(len(self._output_columns or [])),
                "n_features": int(self._backend.n_features),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write the learner, backend state, umbrella and val-slice SHAP.

        - ``learner.json`` + ``learner.ubj`` — the XGBoost model.
        - Backend state (``physchem_scaler.json`` for ``physchem``; nothing
          extra for ``maccs``).
        - ``umbrella.json`` — formula_version, descriptor, output_columns,
          reference_signal, reference_signal_raw.
        - ``reference_self_aggregates.npy`` — sorted val-slice Gini values,
          the calibration CDF.
        - ``val_shap_attributions.npy`` — ``(n_val, n_features)`` float32
          ``|SHAP|`` matrix on the val slice, for offline formula iteration.
        """
        assert self._learner is not None
        assert self._backend is not None
        assert self._sorted_self_aggregates is not None
        self._learner.save(folder)
        self._backend.save_state(folder)
        umbrella_payload = {
            "formula_version": SIGNAL_FORMULA_VERSION,
            "descriptor": self._backend.name,
            "output_columns": list(self._output_columns or []),
            "reference_signal": float(self._reference_signal or 0.0),
            "reference_signal_raw": float(self._reference_signal_raw or 0.0),
        }
        with open(folder / UMBRELLA_FILE, "w") as f:
            json.dump(umbrella_payload, f, indent=2)
        np.save(folder / SELF_AGGREGATES_FILE, self._sorted_self_aggregates)
        if self._val_shap_attributions is not None:
            np.save(folder / VAL_SHAP_ATTRIBUTIONS_FILE, self._val_shap_attributions)

    def _load_own(self, folder: pathlib.Path) -> None:
        """Read the umbrella, verify the formula version, rebuild the backend."""
        umbrella = read_json(folder / UMBRELLA_FILE, self.NAME)
        formula = umbrella.get("formula_version")
        if formula != SIGNAL_FORMULA_VERSION:
            raise ArtifactVersionError(
                f"signal artifact at {folder} was built with formula "
                f"{formula!r}; this eosquality install expects "
                f"{SIGNAL_FORMULA_VERSION!r}. Refit with the current version."
            )
        self._backend = load_backend(umbrella["descriptor"], folder)
        self._learner = SignalLearner.load(folder)
        self._output_columns = list(umbrella["output_columns"])
        self._sorted_self_aggregates = np.load(
            require_file(folder / SELF_AGGREGATES_FILE, self.NAME)
        )
        self._reference_signal = float(umbrella["reference_signal"])
        self._reference_signal_raw = float(umbrella["reference_signal_raw"])
        val_shap_path = folder / VAL_SHAP_ATTRIBUTIONS_FILE
        # Large and only needed for offline analysis: memory-map it.
        self._val_shap_attributions = (
            np.load(val_shap_path, mmap_mode="r") if val_shap_path.is_file() else None
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._learner is not None
            and self._backend is not None
            and self._sorted_self_aggregates is not None
            and self._reference_signal is not None
        )

    @property
    def learner_(self) -> SignalLearner:
        self._check_fitted()
        assert self._learner is not None
        return self._learner

    @property
    def backend_(self) -> DescriptorBackend:
        """The fitted descriptor backend (PhyschemBackend or MaccsBackend)."""
        self._check_fitted()
        assert self._backend is not None
        return self._backend

    @property
    def descriptor_(self) -> str:
        """The descriptor identifier the artifact was trained with."""
        return self.backend_.name

    @property
    def reference_signal_(self) -> float:
        self._check_fitted()
        assert self._reference_signal is not None
        return self._reference_signal

    @property
    def reference_signal_raw_(self) -> float:
        self._check_fitted()
        assert self._reference_signal_raw is not None
        return self._reference_signal_raw

    @property
    def val_shap_attributions_(self) -> np.ndarray | None:
        """``(n_val, n_features)`` float32 ``|SHAP|`` matrix on the val slice.

        Use this for offline experimentation with alternative raw-signal
        formulas without recomputing SHAP.
        """
        self._check_fitted()
        return self._val_shap_attributions

    @property
    def output_columns_(self) -> list[str]:
        self._check_fitted()
        assert self._output_columns is not None
        return list(self._output_columns)
