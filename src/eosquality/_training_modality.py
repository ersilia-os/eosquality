"""Training modality of :class:`~eosquality.quality.ErsiliaQuality`: fit and run.

Module-level functions taking the orchestrator instance ``eq``, like
``_reference_modality.py``, so that ``quality.py`` stays a readable façade.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from eosquality._registry import IN_TRAINING_COLUMN, score_name
from eosquality.scores._error_model import MIN_LABELLED
from eosquality.scores._training_helpers import TrainingQuery
from eosquality.scores.training_difficulty import TrainingDifficulty
from eosquality.scores.training_distance import TrainingDistance
from eosquality.shared.feature_selection import select_by_shared_molecules
from eosquality.training.data import load_training
from eosquality.training.state import fit_training
from eosquality.utils import console
from eosquality.utils.logging import logger


def load_training_sets(training_sets, output_columns=None) -> dict:
    """Load the training sets, in their own console section.

    Parameters
    ----------
    training_sets : str or pathlib.Path
        Folder with one ``<output_column>.csv`` per column.
    output_columns : list of str, optional
        The reference's output columns; every file must name one of them.

    Returns
    -------
    dict of str to TrainingColumn
        The usable columns (at least ``MIN_TRAINING_MOLECULES`` molecules).
    """
    steps = console.Steps(1)
    with console.section("Training sets") as section:
        with steps("Load and standardise the training sets") as st:
            columns = load_training(training_sets, output_columns)
            labelled = sum(c.has_y for c in columns.values())
            st.summary = f"{len(columns)} column(s) · {labelled} with labels"
        console.table(
            ("column", "molecules", "labels", "unparsable", "conflicting"),
            [
                (
                    c.name,
                    f"{c.n:,}",
                    c.y_kind or "—",
                    f"{c.n_unparsable:,}" if c.n_unparsable else "",
                    f"{c.n_conflicting:,}" if c.n_conflicting else "",
                )
                for c in columns.values()
            ],
        )
        _warn_data_issues(columns)
        if output_columns is not None and len(columns) < len(output_columns):
            console.detail(
                [
                    (
                        "reference",
                        f"restricted to these {len(columns)} of "
                        f"{len(output_columns)} output columns",
                    )
                ]
            )
        section.summary = f"{len(columns)} column(s)"
    return columns


def select_training_columns(columns: dict, shared, max_features) -> dict:
    """The training columns to fit: at most ``max_features`` of them.

    With a reference modality they are the reference's selected columns (its
    feature selection ran on the columns with a training set). Training-only,
    the largest training set of each overlap cluster is kept
    (:func:`~eosquality.shared.feature_selection.select_by_shared_molecules`).

    Parameters
    ----------
    columns : dict of str to TrainingColumn
        The usable training sets.
    shared : SharedFitState or None
        The fitted reference shared state, if any.
    max_features : int, optional
        Column cap; ``None`` disables it (training-only fits).

    Returns
    -------
    dict of str to TrainingColumn
        The kept columns, in input order.
    """
    if shared is not None:
        keep = set(shared.selected_columns)
        names = [n for n in columns if n in keep]
    else:
        names = select_by_shared_molecules(
            {n: set(c.smiles) for n, c in columns.items()}, max_features
        )
    if len(names) < len(columns):
        logger.info(
            f"training | {len(names)} of {len(columns)} columns selected: "
            + ", ".join(names)
        )
    return {n: columns[n] for n in names}


def _warn_data_issues(columns: dict) -> None:
    """One warning summing the rows the training-set cleaning changed."""
    bad = sum(c.n_unparsable for c in columns.values())
    conflicting = sum(c.n_conflicting for c in columns.values())
    if bad or conflicting:
        logger.warning(
            f"training sets | {bad:,} unparsable SMILES dropped, {conflicting:,} "
            "molecules with conflicting labels merged (majority for binary labels, "
            "median otherwise); per column in the table above and the log"
        )


def fit_training_modality(
    eq,
    columns: dict,
    *,
    eos_id: str,
    version: str,
    skip: set[str] = frozenset(),
    n_loaded: int | None = None,
) -> None:
    """Fit the training components of ``eq`` on loaded training sets.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator; its training state and components are replaced.
    columns : dict of str to TrainingColumn
        Training sets from :func:`load_training_sets`.
    eos_id, version : str
        Model id and version.
    skip : set of str, optional
        Training components not to fit (``TRAINING_ORDER`` names).
    n_loaded : int, optional
        Number of usable columns before selection, for the console.

    Raises
    ------
    ValueError
        If nothing is left to fit (distance excluded and no labelled column
        for difficulty).
    """
    want_distance = "training_distance" not in skip
    want_difficulty = "training_difficulty" not in skip
    steps = console.Steps(1 + want_distance + want_difficulty)
    with console.section("Training modality") as section:
        if n_loaded is not None and n_loaded > len(columns):
            console.detail(
                [
                    (
                        "columns",
                        f"{len(columns)} of {n_loaded} selected: {', '.join(columns)}",
                    )
                ]
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
        ("column", "labels", "fitted on", "folds", "Spearman (out-of-fold)"),
        [
            (
                n,
                f"{m.n_labelled:,}",
                f"{m.n_fit:,}" if m.n_fit and m.n_fit < m.n_labelled else "all",
                m.cv,
                f"{m.spearman:.3f}",
            )
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
        The training details table (one row per query), or None when
        training distance is not fitted.
    """
    distance, difficulty = eq.training_distance, eq.training_difficulty
    if distance is None and difficulty is None:
        return None
    steps = console.Steps((distance is not None) + (difficulty is not None))
    details = None
    features = TrainingQuery.from_frame(query)
    n_bad = features.n_rows - len(features.smiles)
    if n_bad and eq._shared is None:  # the reference modality already warned
        logger.warning(
            f"{n_bad:,} query row(s) have a missing or unparsable SMILES; their "
            "training scores are NaN."
        )
    with console.section("Training modality"):
        if distance is not None:
            name = score_name("training_distance")
            with steps(f"Score: {name}") as st:
                result = distance.run(query, features)
                st.summary = console.median_summary(result.score)
            columns[name] = result.score
            columns[f"{name}_raw"] = result.score_raw
            metadata.update({f"{name}_{k}": v for k, v in result.metadata.items()})
            details = result.details
        if difficulty is not None:
            name = score_name("training_difficulty")
            with steps(f"Score: {name}") as st:
                scored = difficulty.run(query, features)
                st.summary = console.median_summary(scored.score)
            columns[name] = scored.score
            if scored.inputs is not None:
                # The error model's own inputs, so they can be inspected or
                # modelled directly; prefixed like every other output column.
                for feature in scored.inputs.columns:
                    columns[f"trn_{feature}"] = scored.inputs[feature]
            metadata.update({f"{name}_{k}": v for k, v in scored.metadata.items()})
            if details is not None:
                details.insert(4, name, scored.score.to_numpy())
    if distance is not None:
        columns[IN_TRAINING_COLUMN] = result.in_training
    return details
