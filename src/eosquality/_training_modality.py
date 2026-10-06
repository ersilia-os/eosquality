"""Training modality of :class:`~eosquality.quality.ErsiliaQuality`: fit and run.

Module-level functions taking the orchestrator instance ``eq``, like
``_reference_modality.py``, so that ``quality.py`` stays a readable façade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eosquality._registry import IN_TRAINING_COLUMN, score_name
from eosquality.scores._error_model import MIN_LABELLED
from eosquality.scores.training_difficulty import TrainingDifficulty
from eosquality.scores.training_distance import TrainingDistance
from eosquality.training.data import load_training
from eosquality.training.state import fit_training
from eosquality.utils import console
from eosquality.utils.logging import logger


def fit_training_modality(
    eq,
    training_sets,
    *,
    eos_id: str,
    version: str,
    skip: set[str] = frozenset(),
) -> None:
    """Load the training sets and fit the training components of ``eq``.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator; its training state and components are replaced.
    training_sets : str or pathlib.Path
        Folder with one ``<output_column>.csv`` per column.
    eos_id, version : str
        Model id and version.
    skip : set of str, optional
        Training components not to fit (``TRAINING_ORDER`` names).

    Raises
    ------
    ValueError
        If nothing is left to fit (distance excluded and no labelled column
        for difficulty).
    """
    output_columns = eq._shared.schema.column_names if eq._shared is not None else None
    want_distance = "training_distance" not in skip
    want_difficulty = "training_difficulty" not in skip
    steps = console.Steps(2 + want_distance + want_difficulty)
    with console.section("Training modality") as section:
        with steps("Load the training sets") as st:
            columns = load_training(training_sets, output_columns)
            labelled = sum(c.has_y for c in columns.values())
            st.summary = f"{len(columns)} column(s) · {labelled} with labels"
        console.table(
            ("column", "molecules", "labels"),
            [(c.name, f"{c.n:,}", c.y_kind or "—") for c in columns.values()],
        )
        with steps("Build one Morgan index per column") as st:
            eq._training = fit_training(columns, eos_id=eos_id, version=version)
            st.summary = f"{len(columns)} index(es)"
        eq.training_distance = None
        if want_distance:
            with steps(f"Score: {score_name('training_distance')}") as st:
                eq.training_distance = TrainingDistance().fit(
                    training=eq._training, shared=eq._shared
                )
                st.summary = "leave-one-out tables for 5-NN distances"
        eq.training_difficulty = None
        if want_difficulty:
            _fit_difficulty(eq, steps)
        if eq.training_distance is None and eq.training_difficulty is None:
            raise ValueError(
                "Nothing to fit for the training sets: trn_distance is excluded "
                f"and no column has >= {MIN_LABELLED} labels for trn_difficulty."
            )
        section.summary = f"{len(columns)} column(s)"


def _fit_difficulty(eq, steps) -> None:
    """Fit training difficulty when some column has enough labels."""
    title = f"Score: {score_name('training_difficulty')} (error models)"
    if not TrainingDifficulty.can_fit(eq._training):
        steps.skip(title, f"no column has ≥ {MIN_LABELLED} labels")
        logger.info(
            f"training difficulty | skipped: no column has ≥ {MIN_LABELLED} labels"
        )
        return
    with steps(title) as st:
        eq.training_difficulty = TrainingDifficulty().fit(
            training=eq._training, shared=eq._shared
        )
        models = eq.training_difficulty.models_
        st.summary = f"{len(models)} error model(s)"
    console.table(
        ("column", "labels", "folds", "Spearman (out-of-fold)"),
        [
            (n, f"{m.n_labelled:,}", m.cv, f"{m.spearman:.3f}")
            for n, m in models.items()
        ],
    )


def run_training(
    eq, query: pd.DataFrame, columns: dict[str, pd.Series], metadata: dict[str, Any]
) -> pd.DataFrame | None:
    """Run the fitted training components, filling ``columns`` and ``metadata``.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator.
    query : pandas.DataFrame
        Query with an ``input`` SMILES column.
    columns : dict of str to pandas.Series
        Score columns, updated in place.
    metadata : dict
        Run metadata, updated in place.

    Returns
    -------
    pandas.DataFrame or None
        The training details table (one row per valid query), or None when
        training distance is not fitted.
    """
    distance, difficulty = eq.training_distance, eq.training_difficulty
    if distance is None and difficulty is None:
        return None
    steps = console.Steps((distance is not None) + (difficulty is not None))
    details = None
    with console.section("Training modality"):
        if distance is not None:
            name = score_name("training_distance")
            with steps(f"Score: {name}") as st:
                result = distance.run(query)
                st.summary = f"median {float(result.score.median()):.3f}"
            columns[name] = result.score
            columns[f"{name}_raw"] = result.score_raw
            metadata.update({f"{name}_{k}": v for k, v in result.metadata.items()})
            details = result.details
        if difficulty is not None:
            name = score_name("training_difficulty")
            with steps(f"Score: {name}") as st:
                scored = difficulty.run(query)
                st.summary = f"median {float(scored.score.median()):.3f}"
            columns[name] = scored.score
            metadata.update({f"{name}_{k}": v for k, v in scored.metadata.items()})
            if details is not None:
                # Details rows are the queries with a valid SMILES, in order:
                # exactly those with a finite training distance.
                valid = np.isfinite(result.score.to_numpy())
                details.insert(3, "difficulty", scored.score.to_numpy()[valid])
    if distance is not None:
        columns[IN_TRAINING_COLUMN] = result.in_training
    return details
