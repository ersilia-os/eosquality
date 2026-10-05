"""Training modality of :class:`~eosquality.quality.ErsiliaQuality`: fit and run.

Module-level functions taking the orchestrator instance ``eq``, like
``_reference_modality.py``, so that ``quality.py`` stays a readable façade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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
    training_predictions,
    *,
    eos_id: str,
    version: str,
) -> None:
    """Load the training sets and fit every training component of ``eq``.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator; its training state and components are replaced.
    training_sets : str or pathlib.Path
        Folder with one ``<output_column>.csv`` per column.
    training_predictions : str, pathlib.Path, pandas.DataFrame or None
        The model's predictions on the training molecules.
    eos_id, version : str
        Model id and version.
    """
    output_columns = eq._shared.schema.column_names if eq._shared is not None else None
    steps = console.Steps(4)
    with console.section("Training modality") as section:
        with steps("Load the training sets") as st:
            columns = load_training(training_sets, output_columns, training_predictions)
            labelled = sum(c.has_y for c in columns.values())
            st.summary = f"{len(columns)} column(s) · {labelled} with labels"
        console.table(
            ("column", "molecules", "labels"),
            [(c.name, f"{c.n:,}", c.y_kind or "—") for c in columns.values()],
        )
        with steps("Build one Morgan index per column") as st:
            eq._training = fit_training(columns, eos_id=eos_id, version=version)
            st.summary = f"{len(columns)} index(es)"
        with steps("Training distance") as st:
            eq.training_distance = TrainingDistance().fit(
                training=eq._training, shared=eq._shared
            )
            st.summary = "leave-one-out tables for 5-NN distances"
        eq.training_difficulty = None
        if TrainingDifficulty.can_fit(eq._training):
            with steps("Training difficulty (error models)") as st:
                eq.training_difficulty = TrainingDifficulty().fit(
                    training=eq._training, shared=eq._shared
                )
                models = eq.training_difficulty.models_
                st.summary = f"{len(models)} error model(s)"
            console.table(
                ("column", "labels", "feature set", "Spearman"),
                [
                    (n, f"{m.n_labelled:,}", m.variant, f"{m.spearman:.3f}")
                    for n, m in models.items()
                ],
            )
        else:
            steps.skip(
                "Training difficulty (error models)",
                f"no column has ≥ {MIN_LABELLED} labels",
            )
            logger.info(
                f"training difficulty | skipped: no column has ≥ {MIN_LABELLED} labels"
            )
        section.summary = f"{len(columns)} column(s)"


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
        the training modality is not fitted.
    """
    if eq.training_distance is None:
        return None
    steps = console.Steps(1 + (eq.training_difficulty is not None))
    with console.section("Training modality"):
        with steps("Training distance") as st:
            result = eq.training_distance.run(query)
            st.summary = f"median {float(result.score.median()):.3f}"
        columns["training_distance"] = result.score
        columns["training_distance_raw"] = result.score_raw
        metadata.update(
            {f"training_distance_{k}": v for k, v in result.metadata.items()}
        )
        details = result.details
        if eq.training_difficulty is not None:
            with steps("Training difficulty") as st:
                difficulty = eq.training_difficulty.run(query)
                st.summary = f"median {float(difficulty.score.median()):.3f}"
            columns["training_difficulty"] = difficulty.score
            metadata.update(
                {f"training_difficulty_{k}": v for k, v in difficulty.metadata.items()}
            )
            # Details rows are the queries with a valid SMILES, in order: exactly
            # those with a finite training distance.
            valid = np.isfinite(result.score.to_numpy())
            details.insert(3, "difficulty", difficulty.score.to_numpy()[valid])
    columns["in_training"] = result.in_training
    return details
