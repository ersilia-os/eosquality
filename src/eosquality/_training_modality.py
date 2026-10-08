"""Training modality of :class:`~eosquality.quality.ErsiliaQuality`: fit and run.

Module-level functions taking the orchestrator instance ``eq``, like
``_reference_modality.py``, so that ``quality.py`` stays a readable façade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eosquality._registry import TRAINING_ORDER, score_name
from eosquality.scores._training_helpers import TrainingQuery
from eosquality.scores.training_distance import TrainingDistance
from eosquality.scores.training_match import TrainingMatch
from eosquality.scores.training_physchem import TrainingPhyschem
from eosquality.shared.feature_selection import select_by_shared_molecules
from eosquality.training.data import load_training
from eosquality.training.state import fit_training
from eosquality.utils import console
from eosquality.utils.logging import logger

# Column order of the training details table; columns of an unfitted score are
# simply absent. The scores CSV carries the pct / raw pairs and the two flags;
# the nearest-neighbour table stays in this table.
_DETAILS_ORDER = (
    "key",
    "input",
    "trn_tanimoto_pct",
    "trn_tanimoto_raw",
    "trn_physchem_pct",
    "trn_physchem_raw",
    "trn_physchem_dist",
    "trn_match",
    "trn_scaffold",
    "trn_in_training",
    "nn1_similarity",
    "nn_smiles",
    "nn_keys",
    "nn_similarities",
    "nn_columns",
)


# The scores fitted the same way: from the training state alone, in run order.
_PLAIN_SCORES = {
    "training_distance": TrainingDistance,
    "training_physchem": TrainingPhyschem,
    "training_match": TrainingMatch,
}


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
            st.summary = f"{len(columns)} column(s)"
        console.table(
            ("column", "molecules", "unparsable"),
            [
                (c.name, f"{c.n:,}", f"{c.n_unparsable:,}" if c.n_unparsable else "")
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
    """One warning for the rows the training-set cleaning dropped."""
    bad = sum(c.n_unparsable for c in columns.values())
    if bad:
        logger.warning(
            f"training sets | {bad:,} unparsable SMILES dropped; per column in the "
            "table above and the log"
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
    """
    scores = {name: cls for name, cls in _PLAIN_SCORES.items() if name not in skip}
    steps = console.Steps(1 + len(scores))
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
        with steps("Build a Morgan index per distinct training set") as st:
            eq._training = fit_training(columns, eos_id=eos_id, version=version)
            st.summary = (
                f"{len({id(v) for v in eq._training.indices.values()})} index(es)"
            )
        for name in _PLAIN_SCORES:
            setattr(eq, name, None)
        for name, cls in scores.items():
            with steps(f"Score: {score_name(name)}") as st:
                component = cls().fit(training=eq._training, shared=eq._shared)
                setattr(eq, name, component)
                st.summary = component.fit_summary
        section.summary = f"{len(columns)} column(s)"


def _assemble_details(
    query: pd.DataFrame, details: pd.DataFrame | None, extras: dict[str, Any]
) -> pd.DataFrame | None:
    """Add the per-query values kept out of the scores CSV to the details table.

    Parameters
    ----------
    query : pandas.DataFrame
        The query, in row order (used for ``key`` / ``input`` when training
        distance, which normally supplies them, is not fitted).
    details : pandas.DataFrame or None
        The training distance details, or None when it is not fitted.
    extras : dict
        Column name to per-query values (aligned with ``query``).

    Returns
    -------
    pandas.DataFrame or None
        The table in ``_DETAILS_ORDER``; None when there is nothing to report.
    """
    if details is None:
        if not extras:
            return None
        keys = (
            query["key"].astype(str).tolist()
            if "key" in query.columns
            else [str(i) for i in query.index]
        )
        details = pd.DataFrame({"key": keys, "input": query["input"].tolist()})
    for name, values in extras.items():
        details[name] = np.asarray(values)
    ordered = [c for c in _DETAILS_ORDER if c in details.columns]
    return details[ordered + [c for c in details.columns if c not in ordered]]


def _emit(columns: dict, extras: dict, *series: pd.Series) -> None:
    """Put each Series in the scores columns and in the details extras."""
    for ser in series:
        columns[ser.name] = ser
        extras[ser.name] = ser.to_numpy()


def run_training(
    eq, query: pd.DataFrame, columns: dict[str, pd.Series], metadata: dict[str, Any]
) -> pd.DataFrame | None:
    """Run the fitted training components, filling ``columns`` and ``metadata``.

    ``columns`` receives ``trn_tanimoto_pct`` / ``_raw``, ``trn_physchem_pct`` /
    ``_raw``, ``trn_match`` and ``trn_scaffold``. The returned details table
    repeats them, and adds ``trn_physchem_dist``, the nearest training
    molecules and ``trn_in_training``.

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
        The training details table (one row per query), or None when no
        training score is fitted.
    """
    fitted = [
        (name, getattr(eq, name))
        for name in TRAINING_ORDER
        if getattr(eq, name) is not None
    ]
    if not fitted:
        return None
    steps = console.Steps(len(fitted))
    details = None
    extras: dict[str, Any] = {}
    features = TrainingQuery.from_frame(query)
    n_bad = features.n_rows - len(features.smiles)
    if n_bad and eq._shared is None:  # the reference modality already warned
        logger.warning(
            f"{n_bad:,} query row(s) have a missing or unparsable SMILES; their "
            "training scores are NaN."
        )
    with console.section("Training modality"):
        for component, instance in fitted:
            name = score_name(component)
            with steps(f"Score: {name}") as st:
                result = instance.run(query, features)
                st.summary = (
                    console.share_summary(result.match)
                    if component == "training_match"
                    else console.median_summary(result.score)
                )
            metadata.update({f"{name}_{k}": v for k, v in result.metadata.items()})
            if component == "training_match":
                _emit(columns, extras, result.match, result.scaffold)
            elif component == "training_physchem":
                _emit(columns, extras, result.score, result.score_raw)
                extras[result.distance.name] = result.distance.to_numpy()
            else:
                _emit(columns, extras, result.score, result.score_raw)
                details = result.details
    return _assemble_details(query, details, extras)
