"""ErsiliaQuality: thin orchestrator composing the per-score components.

For users who want a one-stop fit/run interface, this class fits the
shared state once and then each requested score on top. Each score remains
independently saveable / loadable.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd

from eosquality import _artifacts, _reference_modality, _training_modality
from eosquality._registry import (
    ALL_SCORES,
    DEFAULT_MAX_FEATURES,
    LIBRARY_USERS,
    SCORE_ORDER,
    TRAINING_ORDER,
    split_exclude,
)
from eosquality.exceptions import (
    NotFittedError,
    SchemaError,
)
from eosquality.results import RunResult
from eosquality.schema.infer import infer_schema
from eosquality.scores.extremity import Extremity
from eosquality.scores.reference_match import ReferenceMatch
from eosquality.scores.training_distance import TrainingDistance
from eosquality.scores.training_match import TrainingMatch
from eosquality.scores.training_physchem import TrainingPhyschem
from eosquality.scores.typicality import Typicality
from eosquality.shared.state import SharedFitState
from eosquality.training import TrainingFitState
from eosquality.utils import console
from eosquality.utils.identifiers import validate_eos_id, validate_version
from eosquality.utils.logging import logger

__all__ = [
    "ALL_SCORES",
    "ErsiliaQuality",
    "RunResult",
]


class ErsiliaQuality:
    """Orchestrate the quality scores of one Ersilia model.

    Two modalities, each present only if its data was given at fit time:

    - **reference** — the model's predictions on the reference library
      (typicality, extremity, match);
    - **training** — the model's per-output-column training sets
      (training_distance, training_physchem, training_match).
    """

    def __init__(self, verbose: bool = False) -> None:
        """Build an unfitted orchestrator.

        Parameters
        ----------
        verbose : bool, optional
            Show the step-by-step output and DEBUG messages (as the CLI does).
        """
        self.verbose = verbose
        if verbose:
            logger.set_verbosity(True)

        self.typicality: Typicality | None = None
        self.extremity: Extremity | None = None
        self.match: ReferenceMatch | None = None
        self.training_distance: TrainingDistance | None = None
        self.training_physchem: TrainingPhyschem | None = None
        self.training_match: TrainingMatch | None = None
        self._shared: SharedFitState | None = None
        self._training: TrainingFitState | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Fit / run
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame | None = None,
        training_sets: str | pathlib.Path | None = None,
        *,
        eos_id: str,
        version: str = "v1",
        exclude: Iterable[str] = (),
        max_features: int | None = DEFAULT_MAX_FEATURES,
        library: str | pathlib.Path | None = None,
    ) -> ErsiliaQuality:
        """Fit the reference modality, the training modality, or both.

        Every score of each given modality is fitted unless excluded.

        Parameters
        ----------
        reference : pandas.DataFrame, optional
            Predictions on the reference library (``key``, ``input``, one
            numeric column per output), in library order. ``None`` for a
            training-only fit.
        training_sets : str or pathlib.Path, optional
            Folder with one ``<output_column>.csv`` per column. With a
            reference, the reference modality is fitted only on the output
            columns that have a usable training set, ``max_features`` selects
            among them, and the training modality uses the same selection.
        eos_id : str
            Model identifier, e.g. ``"eos4e40"``.
        version : str, optional
            Model version, e.g. ``"v1"``.
        exclude : iterable of str, optional
            Scores not to fit, by public name (``ALL_SCORES``), e.g.
            ``["ref_match"]``.
        max_features : int, optional
            Cap on the output columns used by both modalities; ``None``
            disables it.
        library : str or pathlib.Path, optional
            Reference library folder (default: the resolved canonical
            library). For tests and custom libraries.

        Returns
        -------
        ErsiliaQuality
            ``self``, fitted. See ``docs/api.md`` for details on each argument.
        """
        skip_reference, skip_training = _validate_fit_args(
            reference, training_sets, eos_id, version, exclude
        )
        self._reset()
        t_start = time.perf_counter()
        console.set_active_color(console.STEP_COLORS["fit"])
        logger.info(f"fit | {eos_id} {version}")
        # Training sets first: with a reference, they decide its columns.
        reference, training_columns = _load_training_first(reference, training_sets)
        if reference is not None:
            _reference_modality.fit_reference(
                self,
                reference,
                eos_id=eos_id,
                version=version,
                library=library,
                scores=[c for c in SCORE_ORDER if c not in skip_reference],
                max_features=max_features,
            )
        if training_columns is not None:
            selected = _training_modality.select_training_columns(
                training_columns, self._shared, max_features
            )
            _training_modality.fit_training_modality(
                self,
                selected,
                eos_id=eos_id,
                version=version,
                skip=skip_training,
                n_loaded=len(training_columns),
            )
        self.is_fitted_ = True
        self._log_fit_done(t_start)
        return self

    def run(self, query: pd.DataFrame) -> RunResult:
        """Score query samples against the fitted reference population.

        Validates and scales the query once and passes it to each
        component's :meth:`run`.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference,
            plus an ``'input'`` SMILES column (``'smiles'`` is accepted as an
            alias) if the match scores or a training score was fit.

        Returns
        -------
        RunResult
            See :class:`RunResult` for the column layout and metadata keys.
        """
        self._check_fitted()
        t_start = time.perf_counter()
        components = self._components()
        training_components = self._training_components()
        console.set_active_color(console.STEP_COLORS["run"])
        logger.info(
            f"run | n_query={len(query):,} | scores="
            f"[{', '.join(list(components) + list(training_components))}]"
        )

        needs_input_col = bool(set(components) & LIBRARY_USERS) or bool(
            training_components
        )
        if "input" not in query.columns and "smiles" in query.columns:
            query = query.rename(columns={"smiles": "input"})
        if needs_input_col and "input" not in query.columns:
            raise SchemaError(
                "Query DataFrame must contain an 'input' (or 'smiles') column with "
                "SMILES strings."
            )

        columns: dict[str, pd.Series] = {}
        metadata: dict[str, Any] = {}
        reference_details = None
        if self._shared is not None:
            reference_details = _reference_modality.run_reference(
                self, query, components, columns, metadata
            )

        training_details = _training_modality.run_training(
            self, query, columns, metadata
        )

        scores_df = pd.DataFrame(columns, index=list(query.index))
        _score_table(scores_df)
        means = " · ".join(
            f"{c}={scores_df[c].mean():.3f}"
            for c in scores_df.columns
            if pd.api.types.is_float_dtype(scores_df[c])
        )
        logger.success(
            f"Run complete | {len(scores_df):,} queries | {means or 'no scores'} | "
            f"{time.perf_counter() - t_start:.2f}s"
        )
        return RunResult(
            scores=scores_df,
            metadata=metadata,
            training_details=training_details,
            reference_details=reference_details,
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
            Schema, scaler, selected columns and the scaled reference.
        """
        self._check_fitted()
        assert self._shared is not None
        return self._shared

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_fit_done(self, t_start: float) -> None:
        """Log the ``Fit complete`` line with the fitted scores and the elapsed time."""
        fitted = list(self._components()) + list(self._training_components())
        logger.success(
            f"Fit complete | {len(fitted)} score(s) [{', '.join(fitted)}] | "
            f"{time.perf_counter() - t_start:.2f}s"
        )

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_table(scores: pd.DataFrame) -> None:
    """Print the distribution of each calibrated score (curated output only)."""
    names = [
        c
        for c in scores.columns
        if pd.api.types.is_numeric_dtype(scores[c])
        and not pd.api.types.is_bool_dtype(scores[c])
        and not c.endswith(("_raw", "_log"))
    ]
    console.table(
        ("score", "mean", "median", "min", "max"),
        [
            (
                c,
                *(
                    f"{getattr(scores[c].astype(float), f)():.3f}"
                    for f in ("mean", "median", "min", "max")
                ),
            )
            for c in names
        ],
        title="Score summary",
    )


def _validate_fit_args(
    reference, training_sets, eos_id: str, version: str, exclude
) -> tuple[set[str], set[str]]:
    """Validate the arguments of ``fit``; return the excluded components per modality.

    Parameters
    ----------
    reference : pandas.DataFrame or None
        Reference predictions.
    training_sets : str, pathlib.Path or None
        Training-sets folder.
    eos_id, version : str
        Model identifier and version.
    exclude : iterable of str
        Public score names not to fit.

    Returns
    -------
    tuple of (set of str, set of str)
        Excluded reference components and excluded training components.

    Raises
    ------
    ValueError
        If an argument is invalid, or a given input has no score left to fit.
    """
    validate_eos_id(eos_id)
    validate_version(version)
    if reference is None and training_sets is None:
        raise ValueError("fit needs reference predictions, training sets, or both.")
    skip_reference, skip_training = split_exclude(exclude or ())
    if reference is not None and set(SCORE_ORDER) <= skip_reference:
        raise ValueError("Every reference score is excluded: nothing to fit.")
    if training_sets is not None and set(TRAINING_ORDER) <= skip_training:
        raise ValueError("Every training score is excluded: nothing to fit.")
    return skip_reference, skip_training


def _load_training_first(reference, training_sets):
    """Load the training sets and restrict the reference to their output columns.

    Parameters
    ----------
    reference : pandas.DataFrame or None
        Reference predictions.
    training_sets : str, pathlib.Path or None
        Training-sets folder.

    Returns
    -------
    tuple
        ``(reference, training_columns)``: the reference without the outputs
        that have no usable training set, and the loaded training columns
        (``None`` without training sets).
    """
    if training_sets is None:
        return reference, None
    output_columns = (
        infer_schema(reference).column_names if reference is not None else None
    )
    training_columns = _training_modality.load_training_sets(
        training_sets, output_columns
    )
    if reference is not None:
        reference = _restrict_outputs(reference, output_columns, list(training_columns))
    return reference, training_columns


def _restrict_outputs(
    reference: pd.DataFrame, output_columns: list[str], keep: list[str]
) -> pd.DataFrame:
    """Drop the output columns of ``reference`` that are not in ``keep``.

    Parameters
    ----------
    reference : pandas.DataFrame
        Predictions on the reference library.
    output_columns : list of str
        Its output columns.
    keep : list of str
        The output columns with a usable training set.

    Returns
    -------
    pandas.DataFrame
        ``reference`` without the other output columns (``key`` and
        ``input`` are kept).
    """
    dropped = [c for c in output_columns if c not in set(keep)]
    if dropped:
        logger.info(
            f"fit | reference restricted to the {len(keep)} of "
            f"{len(output_columns)} output columns with a training set"
        )
    return reference.drop(columns=dropped)
