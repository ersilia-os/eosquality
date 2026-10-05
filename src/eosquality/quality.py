"""ErsiliaQuality: thin orchestrator composing the per-score components.

For users who want a one-stop fit/run interface, this class fits the
shared state once, the kNN state once, and then each requested score on
top. Each score remains independently saveable / loadable.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

from eosquality import _artifacts, _reference_modality
from eosquality._registry import (
    ALL_SCORES,
    DEFAULT_SCORES,
    INDEX_AWARE,
    MIN_REFERENCE_SAMPLES,
    SCORE_ORDER,
    TRAINING_ORDER,
)
from eosquality.config import ErsiliaQualityConfig, NeighborConfig
from eosquality.exceptions import (
    NotFittedError,
    SchemaError,
)
from eosquality.results import RunResult
from eosquality.scores._helpers import (
    _resolve_vector_index,
)
from eosquality.scores.consistency import Consistency
from eosquality.scores.extremity import Extremity
from eosquality.scores.signal import Signal
from eosquality.scores.support import Support
from eosquality.scores.training_distance import TrainingDistance
from eosquality.scores.typicality import Typicality
from eosquality.shared.fit import DEFAULT_MAX_FEATURES
from eosquality.shared.state import SharedFitState
from eosquality.training import (
    TrainingFitState,
    fit_training,
    load_training,
)
from eosquality.utils.identifiers import validate_eos_id, validate_version
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

__all__ = [
    "ALL_SCORES",
    "DEFAULT_SCORES",
    "MIN_REFERENCE_SAMPLES",
    "ErsiliaQuality",
    "RunResult",
]


class ErsiliaQuality:
    """Orchestrate the quality scores of one Ersilia model.

    Two modalities, each present only if its data was given at fit time:

    - **reference** — the model's predictions on the reference library
      (typicality, extremity, support, consistency, signal);
    - **training** — the model's per-output-column training sets
      (training_distance).
    """

    def __init__(
        self,
        k: int = 5,
        verbose: bool = False,
        config: ErsiliaQualityConfig | None = None,
    ) -> None:
        """Build an unfitted orchestrator.

        Parameters
        ----------
        k:
            Number of nearest neighbors for the FP self-kNN. Ignored
            if ``config`` is provided.
        verbose:
            If ``True``, route the package's loguru output to stderr at
            DEBUG level.
        config:
            Full :class:`ErsiliaQualityConfig`. If omitted, a default
            config is built from ``k``.
        """
        if config is not None:
            self.config = config
        else:
            self.config = ErsiliaQualityConfig(neighbors=NeighborConfig(k=k))
        self.verbose = verbose
        if verbose:
            logger.set_verbosity(True)

        self.typicality: Typicality | None = None
        self.support: Support | None = None
        self.consistency: Consistency | None = None
        self.extremity: Extremity | None = None
        self.signal: Signal | None = None
        self.training_distance: TrainingDistance | None = None
        self._shared: SharedFitState | None = None
        self._training: TrainingFitState | None = None
        self._vector_index_cache: VectorIndex | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Fit / run
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame | None = None,
        eos_id: str | None = None,
        version: str = "v1",
        vector_index: str | pathlib.Path | None = None,
        ignore_size: bool = False,
        scores: Iterable[str] = DEFAULT_SCORES,
        max_features: int | None = DEFAULT_MAX_FEATURES,
        max_signal_train_samples: int | None = 1000,
        signal_descriptor: str = "physchem",
        training_sets: str | pathlib.Path | None = None,
        training_predictions: str | pathlib.Path | pd.DataFrame | None = None,
    ) -> ErsiliaQuality:
        """Fit the reference modality, the training modality, or both.

        Parameters
        ----------
        reference : pandas.DataFrame, optional
            Predictions on the reference library (``key``, ``input``, one
            numeric column per output). ``None`` for a training-only fit.
        eos_id : str
            Model identifier, e.g. ``"eos4e40"``.
        version : str, optional
            Dataset version, e.g. ``"v1"``.
        vector_index : str or pathlib.Path, optional
            Custom index folder for the reference modality (default: canonical).
        ignore_size : bool, optional
            Skip the ``MIN_REFERENCE_SAMPLES`` minimum (testing only).
        scores : iterable of str, optional
            Reference scores to fit, from ``ALL_SCORES``.
        max_features : int, optional
            Feature-selection cap; ``None`` disables it.
        max_signal_train_samples : int, optional
            Signal training rows; ``None`` or ``0`` uses the full train slice.
        signal_descriptor : {"physchem", "maccs"}, optional
            Feature backend of the signal score.
        training_sets : str or pathlib.Path, optional
            Folder with one ``<output_column>.csv`` per column; fits the training mode.
        training_predictions : str, pathlib.Path or pandas.DataFrame, optional
            The model's predictions on the training molecules.

        Returns
        -------
        ErsiliaQuality
            ``self``, fitted. See ``docs/api.md`` for details on each argument.
        """
        if eos_id is None:
            raise ValueError("fit needs eos_id= (e.g. 'eos4e40').")
        validate_eos_id(eos_id)
        validate_version(version)
        if reference is None and training_sets is None:
            raise ValueError("fit needs reference predictions, training sets, or both.")
        self._reset()
        t_start = time.perf_counter()
        logger.rule(f"ErsiliaQuality · fit · {eos_id} {version}")
        if reference is not None:
            _reference_modality.fit_reference(
                self,
                reference,
                eos_id=eos_id,
                version=version,
                vector_index=vector_index,
                ignore_size=ignore_size,
                scores=scores,
                max_features=max_features,
                max_signal_train_samples=max_signal_train_samples,
                signal_descriptor=signal_descriptor,
            )
        if training_sets is not None:
            self.fit_training(
                training_sets, training_predictions, eos_id=eos_id, version=version
            )
        self.is_fitted_ = True
        fitted = list(self._components()) + list(self._training_components())
        logger.success(
            f"Fit complete | {len(fitted)} score(s) [{', '.join(fitted)}] | "
            f"{time.perf_counter() - t_start:.2f}s"
        )
        logger.rule()
        return self

    def fit_training(
        self,
        training_sets: str | pathlib.Path,
        training_predictions: str | pathlib.Path | pd.DataFrame | None = None,
        *,
        eos_id: str | None = None,
        version: str | None = None,
    ) -> ErsiliaQuality:
        """Fit (or replace) the training modality on this instance.

        Works on a fresh instance (training-only), after a reference fit, or on
        an instance loaded from artifacts (adding training later). With a
        reference modality, training files must name its output columns and
        the model id must match.

        Parameters
        ----------
        training_sets : str or pathlib.Path
            Folder with one ``<output_column>.csv`` per column.
        training_predictions : str, pathlib.Path or pandas.DataFrame, optional
            The model's predictions on the training molecules.
        eos_id, version : str, optional
            Model id; default to the reference modality's.

        Returns
        -------
        ErsiliaQuality
            ``self``, with the training modality fitted.
        """
        known_id, known_version = self._model_id()
        if known_id and eos_id and eos_id != known_id:
            raise ValueError(
                f"Training sets are for {eos_id} but the artifacts are for {known_id}."
            )
        eos_id = eos_id or known_id
        version = version or known_version or "v1"
        if not eos_id:
            raise ValueError("fit_training needs eos_id= for a training-only fit.")
        validate_eos_id(eos_id)
        validate_version(version)
        t = time.perf_counter()
        output_columns = (
            self._shared.schema.column_names if self._shared is not None else None
        )
        columns = load_training(training_sets, output_columns, training_predictions)
        logger.info(f"training | {len(columns)} column(s) loaded | building indices…")
        self._training = fit_training(columns, eos_id=eos_id, version=version)
        self.training_distance = TrainingDistance().fit(
            training=self._training, shared=self._shared
        )
        self.is_fitted_ = True
        logger.info(f"training | fitted | {time.perf_counter() - t:.1f}s")
        return self

    def run(self, query: pd.DataFrame) -> RunResult:
        """Score query samples against the fitted reference population.

        Validates and scales the query once, computes the FP-selected kNN
        once for Support + Consistency, and passes the precomputed arrays
        to each component's :meth:`run`.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference,
            plus an ``'input'`` SMILES column if Support, Consistency or
            Signal was fit.

        Returns
        -------
        RunResult
            See :class:`RunResult` for the column layout and metadata keys.
        """
        self._check_fitted()
        t_start = time.perf_counter()
        components = self._components()
        training_components = self._training_components()
        logger.rule(f"ErsiliaQuality · run · {len(query):,} queries")
        logger.info(
            f"run | n_query={len(query):,} | scores="
            f"[{', '.join(list(components) + list(training_components))}]"
        )

        needs_input_col = bool(set(components) & INDEX_AWARE) or bool(
            training_components
        )
        if needs_input_col and "input" not in query.columns:
            raise SchemaError(
                "Query DataFrame must contain an 'input' column with SMILES strings."
            )

        columns: dict[str, pd.Series] = {}
        metadata: dict[str, Any] = {}
        if self._shared is not None:
            _reference_modality.run_reference(
                self, query, components, columns, metadata
            )

        training_details = None
        if self.training_distance is not None:
            t = time.perf_counter()
            result = self.training_distance.run(query)
            columns["training_distance"] = result.score
            columns["training_distance_raw"] = result.score_raw
            columns["training_n_columns"] = result.n_columns
            columns["in_training_any"] = result.in_training_any
            metadata.update(
                {f"training_distance_{k}": v for k, v in result.metadata.items()}
            )
            training_details = result.details
            logger.info(
                f"score 'training_distance' | median={float(result.score.median()):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        scores_df = pd.DataFrame(columns, index=list(query.index))
        logger.scores_summary_table(scores_df)
        means = " · ".join(
            f"{c}={scores_df[c].mean():.3f}"
            for c in scores_df.columns
            if pd.api.types.is_float_dtype(scores_df[c])
        )
        logger.success(
            f"Run complete | {len(scores_df):,} queries | {means or 'no scores'} | "
            f"{time.perf_counter() - t_start:.2f}s"
        )
        logger.rule()
        return RunResult(
            scores=scores_df, metadata=metadata, training_details=training_details
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write the fitted artifacts to a folder.

        Parameters
        ----------
        path : str or pathlib.Path
            Artifacts folder (created if needed).

        Returns
        -------
        pathlib.Path
            The folder written.
        """
        return _artifacts.save(self, path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> ErsiliaQuality:
        """Reconstruct an orchestrator from a saved artifacts folder.

        Parameters
        ----------
        path : str or pathlib.Path
            Folder written by :meth:`save`.

        Returns
        -------
        ErsiliaQuality
            A fitted instance with every component found in the folder.
        """
        return _artifacts.load(cls, path)

    @classmethod
    def add_training(
        cls,
        path: str | pathlib.Path,
        training_sets: str | pathlib.Path,
        training_predictions: str | pathlib.Path | pd.DataFrame | None = None,
        *,
        eos_id: str | None = None,
        version: str | None = None,
    ) -> ErsiliaQuality:
        """Add the training modality to an existing artifacts folder in place.

        Parameters
        ----------
        path : str or pathlib.Path
            Existing artifacts folder (must not already hold a training modality).
        training_sets : str or pathlib.Path
            Folder with one ``<output_column>.csv`` per column.
        training_predictions : str, pathlib.Path or pandas.DataFrame, optional
            The model's predictions on the training molecules.
        eos_id, version : str, optional
            Model id of the training sets; must match the artifacts' model.

        Returns
        -------
        ErsiliaQuality
            The loaded instance with the training modality added.
        """
        return _artifacts.add_training(
            cls,
            path,
            training_sets,
            training_predictions,
            eos_id=eos_id,
            version=version,
        )

    # ------------------------------------------------------------------
    # Post-fit attributes
    # ------------------------------------------------------------------

    @property
    def schema_(self):
        """Schema inferred from the reference predictions.

        Returns
        -------
        Schema
            Reference modality only.
        """
        self._check_fitted()
        assert self._shared is not None
        return self._shared.schema

    @property
    def reference_support_(self) -> float:
        """Mean calibrated support of the reference molecules (about 0.5).

        Returns
        -------
        float
            Requires the support score to be fitted.
        """
        return self._anchor("support")

    @property
    def reference_typicality_(self) -> float:
        """Mean calibrated typicality of the reference molecules (about 0.5).

        Returns
        -------
        float
            Requires the typicality score to be fitted.
        """
        return self._anchor("typicality")

    @property
    def reference_extremity_(self) -> float:
        """Mean calibrated extremity of the reference molecules (about 0.5).

        Returns
        -------
        float
            Requires the extremity score to be fitted.
        """
        return self._anchor("extremity")

    @property
    def reference_consistency_(self) -> float:
        """Mean calibrated consistency of the reference molecules (about 0.5).

        Returns
        -------
        float
            Requires the consistency score to be fitted.
        """
        return self._anchor("consistency")

    @property
    def reference_signal_(self) -> float:
        """Mean calibrated signal of the reference molecules (about 0.5).

        Returns
        -------
        float
            Requires the signal score to be fitted.
        """
        return self._anchor("signal")

    @property
    def modalities_(self) -> list[str]:
        """Fitted modalities.

        Returns
        -------
        list of str
            ``["reference"]``, ``["training"]`` or both.
        """
        self._check_fitted()
        return [
            m
            for m, present in (
                ("reference", self._shared is not None),
                ("training", self._training is not None),
            )
            if present
        ]

    @property
    def metadata_(self):
        """Provenance and dataset statistics of the reference fit.

        Returns
        -------
        FitMetadata
            Reference modality only.
        """
        self._check_fitted()
        assert self._shared is not None
        return self._shared.metadata

    @property
    def shared_(self) -> SharedFitState:
        """Shared fit state of the reference modality.

        Returns
        -------
        SharedFitState
            Schema, scaler, splits, selected columns and the scaled reference.
        """
        self._check_fitted()
        assert self._shared is not None
        return self._shared

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise NotFittedError(
                "This ErsiliaQuality instance is not fitted yet. Call fit() first."
            )

    def _anchor(self, score: str) -> float:
        """Reference anchor of a fitted reference score (raises if not fitted)."""
        self._check_fitted()
        component = getattr(self, score)
        if component is None:
            raise RuntimeError(
                f"reference_{score} is only defined when {score} has been fit."
            )
        return getattr(component, f"reference_{score}_")

    def _reset(self) -> None:
        """Drop every fitted component (a re-fit replaces all of them)."""
        for name in SCORE_ORDER + TRAINING_ORDER:
            setattr(self, name, None)
        self._shared = None
        self._training = None
        self._vector_index_cache = None
        self.is_fitted_ = False

    def _components(self) -> dict[str, Any]:
        """Fitted reference-modality components keyed by name, in canonical order."""
        return {
            name: getattr(self, name)
            for name in SCORE_ORDER
            if getattr(self, name) is not None
        }

    def _training_components(self) -> dict[str, Any]:
        """Fitted training-modality components keyed by name."""
        return {
            name: getattr(self, name)
            for name in TRAINING_ORDER
            if getattr(self, name) is not None
        }

    def _model_id(self) -> tuple[str, str]:
        """``(eos_id, version)`` from whichever modality is fitted ("" if none)."""
        if self._shared is not None:
            return self._shared.metadata.eos_id, self._shared.metadata.version
        if self._training is not None:
            return self._training.eos_id, self._training.version
        return "", ""

    def _get_vector_index(self) -> VectorIndex:
        """Load (and cache) the VectorIndex backing the index-aware scores.

        See :func:`eosquality.scores._helpers._resolve_vector_index`.
        """
        if self._vector_index_cache is not None:
            return self._vector_index_cache
        assert self._shared is not None
        self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
