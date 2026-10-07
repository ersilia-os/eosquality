"""Reference modality of :class:`~eosquality.quality.ErsiliaQuality`: fit and run.

Module-level functions taking the orchestrator instance ``eq``; the class
delegates to them so that ``quality.py`` stays a readable façade.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from typing import Any

import pandas as pd

from eosquality._registry import SCORE_ORDER, score_name
from eosquality.exceptions import SchemaError
from eosquality.library.reference import ReferenceLibrary
from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import _make_query_repr
from eosquality.scores.extremity import Extremity
from eosquality.scores.reference_match import ReferenceMatch
from eosquality.scores.typicality import Typicality
from eosquality.shared.fit import fit_shared
from eosquality.utils import console
from eosquality.utils.logging import logger


def fit_reference(
    eq,
    reference: pd.DataFrame,
    *,
    eos_id: str,
    version: str,
    library: str | pathlib.Path | None,
    scores: Iterable[str],
    max_features: int | None,
) -> None:
    """Fit the reference-modality components of ``eq`` on the reference predictions.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator to fill (``_shared``, the score attributes).
    reference : pandas.DataFrame
        Predictions on the reference library, in library order.
    eos_id, version : str
        Model identifier and dataset version.
    library : str, pathlib.Path or None
        Custom reference-library folder, or ``None`` for the resolved
        canonical library.
    scores : iterable of str
        Reference components to fit (``SCORE_ORDER`` names).
    max_features : int or None
        Feature-selection cap.
    """
    scores_set = set(scores)
    requested = [n for n in SCORE_ORDER if n in scores_set]
    steps = console.Steps(3 + len(requested))
    with console.section("Reference modality") as section:
        with steps("Validate reference predictions") as st:
            _validate_reference(reference)
            st.summary = f"{len(reference):,} molecules · {len(requested)} score(s)"
        logger.info(
            f"fit | eos_id={eos_id} version={version} scores=[{', '.join(requested)}]"
        )
        with steps("Load the reference library") as st:
            lib = ReferenceLibrary.load(library)
            lib.validate_smiles(list(reference["input"]))
            st.summary = (
                f"{console.plain(lib.library_name or 'custom library')} · "
                f"{lib.n_reference:,} molecules match the reference"
            )
        with steps("Shared state: schema, scaling, feature selection") as st:
            shared = fit_shared(
                reference,
                eos_id=eos_id,
                version=version,
                library_id=lib.library_name,
                library_path=str(lib.path.resolve()) if library else "",
                max_features=max_features,
            )
            st.summary = (
                f"{len(shared.schema.columns)} output(s) → "
                f"{len(shared.selected_columns)} selected"
            )
        eq._shared = shared
        fitters = _fitters(reference, shared, lib)
        for name in requested:
            with steps(f"Score: {score_name(name)}"):
                setattr(eq, name, fitters[name]())
                anchor = getattr(getattr(eq, name), "anchor_", None)
                logger.info(f"score {name!r} | fitted | reference={anchor}")
        section.summary = f"{len(requested)} score(s) fitted"


def _fitters(reference, shared, library):
    """Zero-argument fitters for each reference score."""
    return {
        "typicality": lambda: Typicality().fit(reference, shared=shared),
        "extremity": lambda: Extremity().fit(reference, shared=shared),
        "match": lambda: ReferenceMatch().fit(
            reference, shared=shared, library=library
        ),
    }


def _validate_reference(reference: pd.DataFrame) -> None:
    """Check the reference table: non-empty, SMILES input column, unique keys."""
    if reference.empty:
        raise SchemaError("Reference DataFrame is empty.")
    validate_input_column(reference)
    check_unique_keys(reference)


def run_reference(
    eq,
    query: pd.DataFrame,
    components: dict[str, Any],
    columns: dict[str, pd.Series],
    metadata: dict[str, Any],
) -> pd.DataFrame | None:
    """Run the reference-modality components, filling ``columns``/``metadata``.

    Parameters
    ----------
    eq : ErsiliaQuality
        The fitted orchestrator.
    query : pandas.DataFrame
        Query predictions.
    components : dict
        Fitted reference components, by name, in canonical order.
    columns : dict of str to pandas.Series
        Output score columns; filled in place.
    metadata : dict
        Run metadata; filled in place.

    Returns
    -------
    pandas.DataFrame or None
        The reference details table (per-column typicality and extremity, one
        row per query), or None when neither is fitted.
    """
    assert eq._shared is not None
    results: dict[str, Any] = {}
    steps = console.Steps(1 + len(components))
    with console.section("Reference modality") as section:
        with steps("Validate and scale the query") as st:
            validate_against_schema(query, eq._shared.schema)
            query_repr = _make_query_repr(eq._shared, query)
            st.summary = (
                f"{query_repr.shape[0]:,} molecules · {query_repr.shape[1]} feature(s)"
            )
        metadata["n_reference"] = len(eq._shared.reference_ids)
        for name, component in components.items():
            column = score_name(name)
            with steps(f"Score: {column}") as st:
                result = _run_component(name, component, query, query_repr)
                st.summary = (
                    console.share_summary(result.match)
                    if name == "match"
                    else console.median_summary(result.score)
                )
            if name == "match":
                columns[result.match.name] = result.match
                columns[result.scaffold.name] = result.scaffold
            else:
                columns[result.score.name] = result.score
                columns[result.score_raw.name] = result.score_raw
                results[name] = result
                logger.info(
                    f"score {name!r} | mean={float(result.score.mean()):.4f} "
                    f"raw mean={float(result.score_raw.mean()):.4f}"
                )
            metadata.update({f"{column}_{k}": v for k, v in result.metadata.items()})
        section.summary = f"{len(components)} score(s)"
    return _reference_details(query, results) if results else None


def _reference_details(query: pd.DataFrame, results: dict[str, Any]) -> pd.DataFrame:
    """Per-column values of each query, ``<column>_<score>_raw`` / ``_pct``.

    Parameters
    ----------
    query : pandas.DataFrame
        Query predictions.
    results : dict
        Typicality and/or extremity run results, by component name.

    Returns
    -------
    pandas.DataFrame
        ``key``, ``input`` (when given), then per score and column the raw
        value and the percentile on that column's reference distribution.
    """
    keys = (
        query["key"].astype(str).tolist()
        if "key" in query.columns
        else [str(i) for i in query.index]
    )
    parts = {"key": keys}
    if "input" in query.columns:
        parts["input"] = query["input"].tolist()
    for name, result in results.items():
        for column in result.per_feature.columns:
            parts[f"{column}_{name}_raw"] = result.per_feature[column].to_numpy()
            parts[f"{column}_{name}_pct"] = result.per_feature_pct[column].to_numpy()
    return pd.DataFrame(parts)


def _run_component(name, component, query, query_repr):
    """Run one reference component with the precomputed shared inputs."""
    if name in ("typicality", "extremity"):
        return component.run(query, query_repr=query_repr)
    return component.run(query)


def validate_input_column(reference: pd.DataFrame) -> None:
    """Require a non-empty ``input`` SMILES column without NaN.

    Parameters
    ----------
    reference : pandas.DataFrame
        The reference predictions.
    """
    if "input" not in reference.columns:
        raise SchemaError(
            "Reference DataFrame must contain an 'input' column with SMILES "
            "strings, to check it against the reference library."
        )
    null_smiles = reference["input"].isna()
    if null_smiles.any():
        raise SchemaError(
            f"Reference 'input' column has {int(null_smiles.sum())} NaN value(s). "
            "All SMILES must be valid strings."
        )
    empty_smiles = reference["input"] == ""
    if empty_smiles.any():
        raise SchemaError(
            f"Reference 'input' column has {int(empty_smiles.sum())} empty string(s). "
            "All SMILES must be non-empty."
        )


def check_unique_keys(reference: pd.DataFrame) -> None:
    """Require unique values in the ``key`` column, if present.

    Parameters
    ----------
    reference : pandas.DataFrame
        The reference predictions.
    """
    if "key" not in reference.columns:
        return
    keys = reference["key"].astype(str)
    dupes = keys[keys.duplicated()]
    if len(dupes):
        n_shown = min(5, len(dupes))
        examples = ", ".join(
            f"row {i}: {k!r}" for i, k in list(dupes.items())[:n_shown]
        )
        raise ValueError(
            f"Duplicate keys in reference: {len(dupes)} duplicate(s). "
            f"First {n_shown}: [{examples}]. "
            "The 'key' column must be unique across rows."
        )
