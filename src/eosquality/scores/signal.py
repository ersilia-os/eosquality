"""Signal score: per-query SHAP-attribution-Gini on a chemical descriptor.

Train one XGBoost regressor on the reference library and reduce
per-query ``|SHAP|`` attributions to a Gini coefficient: high when
attribution is focused on a few features ("focused" chemistry), low
when attribution scatters across many features ("scattered"
chemistry).

The features are the RDKit physicochemical descriptor set
(``Descriptors._descList``; ~200 descriptors, exact count depends on the
RDKit version), precomputed at library build time alongside the FP index.
The descriptor is recorded in ``umbrella.json``. The learner trains on
``TRAIN_SAMPLES`` rows of the canonical train slice.

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

from eosquality.exceptions import ArtifactVersionError
from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._descriptors import (
    DEFAULT_DESCRIPTOR,
    DESCRIPTOR_NAMES,
    DescriptorBackend,
    load_backend,
    make_backend,
)
from eosquality.scores._helpers import (
    _parses,
    _reference_repr,
    _score_from_aggregates,
)
from eosquality.scores._signal_learner import (
    SignalLearner,
    _shap_attributions,
    _shap_signal_raw,
    _signal_raw_from_attributions,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

SUBFOLDER = "signal"
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
# Rows of the train slice the learner is trained on (calibration always uses
# the full val slice).
TRAIN_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _backend_for(descriptor: str, vector_index) -> DescriptorBackend:
    """Fit-time descriptor backend over the library behind ``vector_index``."""
    if descriptor not in DESCRIPTOR_NAMES:
        raise ValueError(
            f"Unknown signal descriptor {descriptor!r}; expected one of "
            f"{DESCRIPTOR_NAMES}."
        )
    vi = (
        vector_index
        if isinstance(vector_index, VectorIndex)
        else VectorIndex.load(pathlib.Path(vector_index))
    )
    return make_backend(descriptor, vi)


def _calibrate(learner, X_val: np.ndarray):
    """Val-slice |SHAP| matrix, per-row Gini and the sorted Gini CDF table."""
    val_attribution = _shap_attributions(learner.model_, X_val)
    ref_agg = _signal_raw_from_attributions(val_attribution)
    return val_attribution, ref_agg, np.sort(ref_agg).astype(np.float64)


def _clean_split(
    Y: np.ndarray, shared: SharedFitState, max_train_samples: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Train (capped) and val row indices whose selected outputs are all finite."""
    non_nan = ~np.isnan(Y).any(axis=1)
    train_idx = _train_subset(shared.splits.train_indices, max_train_samples)
    train_idx = train_idx[non_nan[train_idx]]
    val_idx = shared.splits.val_indices[non_nan[shared.splits.val_indices]]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            "Signal.fit needs at least one train and one val row whose selected "
            f"outputs are all finite (got n_train={len(train_idx)}, "
            f"n_val={len(val_idx)})."
        )
    return train_idx, val_idx


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
    reference library, using RDKit physchem descriptors. At run time, for each
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
        **learner_kwargs: Any,
    ) -> Signal:
        """Fit the XGBoost regressor and calibrate its SHAP Gini on the val slice.

        Parameters
        ----------
        reference : pandas.DataFrame
            Predictions on the reference library.
        vector_index : str, pathlib.Path or VectorIndex
            Index whose folder holds the library descriptor matrices.
        shared : SharedFitState
            Shared fit state (splits, scaled outputs, selected columns).
        **learner_kwargs
            Forwarded to :meth:`SignalLearner.fit_from_arrays`.

        Returns
        -------
        Signal
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        backend = _backend_for(DEFAULT_DESCRIPTOR, vector_index)
        Y, cols = _normalize_y(reference, shared)
        train_idx, val_idx = _clean_split(Y, shared, TRAIN_SAMPLES)
        logger.info(
            f"signal | fit | descriptor={backend.name} n_features={backend.n_features} "
            f"n_train={len(train_idx):,} n_val={len(val_idx):,}"
        )
        X_val = backend.compute_reference_subset(reference, val_idx).astype(
            np.float32, copy=False
        )
        learner = SignalLearner().fit_from_arrays(
            X_train=backend.compute_reference_subset(reference, train_idx).astype(
                np.float32, copy=False
            ),
            Y_train=Y[train_idx],
            X_val=X_val,
            Y_val=Y[val_idx],
            output_columns=cols,
            **learner_kwargs,
        )
        val_attribution, ref_agg, sorted_self = _calibrate(learner, X_val)

        self._shared = shared
        self._learner = learner
        self._output_columns = list(shared.selected_columns)
        self._backend = backend
        self._sorted_self_aggregates = sorted_self
        self._reference_signal = float(
            np.mean(_score_from_aggregates(ref_agg, sorted_self))
        )
        self._reference_signal_raw = float(ref_agg.mean())
        self._val_shap_attributions = val_attribution.astype(np.float32)
        self._finish_fit(t0)
        logger.info(
            f"signal | fitted | reference_signal={self._reference_signal:.4f} "
            f"gini median={float(np.median(ref_agg)):.4f} | "
            f"{self._fit_duration_seconds:.1f}s"
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

        Returns
        -------
        SignalRunResult
            Calibrated score, raw Gini and metadata.
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
        # Unparsable SMILES would become placeholder descriptor rows (zeros or
        # NaN) and get a meaningless score; they are NaN instead.
        valid = np.array([_parses(s) for s in smiles_list], dtype=bool)
        row_aggregate = np.full(len(smiles_list), np.nan)
        if valid.any():
            query_X = self._backend.query_matrix(
                [s for s, ok in zip(smiles_list, valid, strict=True) if ok]
            ).astype(np.float32, copy=False)
            row_aggregate[valid] = _shap_signal_raw(self._learner.model_, query_X)
        score = _score_from_aggregates(row_aggregate, self._sorted_self_aggregates)
        logger.debug(
            f"signal | run | {len(smiles_list):,} queries | "
            f"descriptor={self._backend.name} | {time.perf_counter() - t0:.1f}s"
        )
        idx = list(query.index)
        return SignalRunResult(
            score=pd.Series(score, index=idx, name="ref_signal"),
            score_raw=pd.Series(row_aggregate, index=idx, name="ref_signal_raw"),
            metadata={
                "descriptor": self._backend.name,
                "anchor": self._reference_signal,
                "anchor_raw": self._reference_signal_raw,
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
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return (
            self._shared is not None
            and self._learner is not None
            and self._backend is not None
            and self._sorted_self_aggregates is not None
            and self._reference_signal is not None
        )

    @property
    def learner_(self) -> SignalLearner:
        """The fitted XGBoost learner.

        Returns
        -------
        SignalLearner
        """
        self._check_fitted()
        assert self._learner is not None
        return self._learner

    @property
    def backend_(self) -> DescriptorBackend:
        """The fitted descriptor backend.

        Returns
        -------
        PhyschemBackend
        """
        self._check_fitted()
        assert self._backend is not None
        return self._backend

    @property
    def descriptor_(self) -> str:
        """The descriptor identifier the artifact was trained with.

        Returns
        -------
        str
            ``"physchem"``.
        """
        return self.backend_.name

    @property
    def reference_signal_(self) -> float:
        """Mean calibrated signal of the val slice (about 0.5).

        Returns
        -------
        float
        """
        self._check_fitted()
        assert self._reference_signal is not None
        return self._reference_signal

    @property
    def reference_signal_raw_(self) -> float:
        """Mean raw Gini of the val slice.

        Returns
        -------
        float
        """
        self._check_fitted()
        assert self._reference_signal_raw is not None
        return self._reference_signal_raw

    @property
    def val_shap_attributions_(self) -> np.ndarray | None:
        """``(n_val, n_features)`` float32 ``|SHAP|`` matrix on the val slice.

        Use this for offline experimentation with alternative raw-signal
        formulas without recomputing SHAP.

        Returns
        -------
        numpy.ndarray or None
        """
        self._check_fitted()
        return self._val_shap_attributions

    @property
    def output_columns_(self) -> list[str]:
        """Output columns the learner was trained on.

        Returns
        -------
        list of str
        """
        self._check_fitted()
        assert self._output_columns is not None
        return list(self._output_columns)
