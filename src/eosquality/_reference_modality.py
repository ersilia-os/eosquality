"""Reference modality of :class:`~eosquality.quality.ErsiliaQuality`: fit and run.

Module-level functions taking the orchestrator instance ``eq``; the class
delegates to them so that ``quality.py`` stays a readable façade.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from eosquality._registry import (
    KNN_USERS,
    N_NEIGHBORS,
    SCORE_ORDER,
    score_name,
)
from eosquality.exceptions import SchemaError
from eosquality.knn.fit import fit_knn
from eosquality.library.identity import reference_library_path
from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import (
    _custom_index_path,
    _make_query_repr,
    _query_fp_distances,
    _query_output_distances,
)
from eosquality.scores.consistency import Consistency
from eosquality.scores.extremity import Extremity
from eosquality.scores.signal import Signal
from eosquality.scores.support import Support
from eosquality.scores.typicality import Typicality
from eosquality.shared.fit import fit_shared
from eosquality.utils import console
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex


def fit_reference(
    eq,
    reference: pd.DataFrame,
    *,
    eos_id: str,
    version: str,
    vector_index: str | pathlib.Path | None,
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
    vector_index : str, pathlib.Path or None
        Custom index folder, or ``None`` for the resolved canonical library.
    scores : iterable of str
        Reference components to fit (``SCORE_ORDER`` names).
    max_features : int or None
        Feature-selection cap.
    """
    scores_set = set(scores)
    requested = [n for n in SCORE_ORDER if n in scores_set]
    uses_knn = bool(scores_set & KNN_USERS)
    steps = console.Steps(3 + uses_knn + len(requested))
    with console.section("Reference modality") as section:
        with steps("Validate reference predictions") as st:
            _validate_reference(reference)
            st.summary = f"{len(reference):,} molecules · {len(requested)} score(s)"
        logger.info(
            f"fit | eos_id={eos_id} version={version} scores=[{', '.join(requested)}]"
        )
        vi, shared, knn = _fit_upstream(
            eq,
            reference,
            steps,
            eos_id=eos_id,
            version=version,
            vector_index=vector_index,
            max_features=max_features,
            uses_knn=uses_knn,
        )
        fitters = _fitters(reference, shared, vi, knn)
        for name in requested:
            with steps(f"Score: {score_name(name)}"):
                setattr(eq, name, fitters[name]())
                anchor = getattr(getattr(eq, name), f"reference_{name}_", None)
                logger.info(f"score {name!r} | fitted | reference={anchor}")
        section.summary = f"{len(requested)} score(s) fitted"
    eq._vector_index_cache = vi


def _fit_upstream(
    eq, reference, steps, *, eos_id, version, vector_index, max_features, uses_knn
):
    """Index, shared state and kNN steps; return ``(vi, shared, knn)``.

    The index is always loaded: it checks that the reference predictions are
    for exactly the library's molecules, in order (which also fixes their
    number).
    """
    with steps("Load the reference library") as st:
        vi = _load_index(reference, vector_index)
        st.summary = (
            f"{console.plain(vi.library_name or 'custom index')} · "
            f"{vi.n_reference:,} molecules match the reference"
        )
    with steps("Shared state: schema, scaling, feature selection, splits") as st:
        shared = fit_shared(
            reference,
            eos_id=eos_id,
            version=version,
            library_id=vi.library_name,
            vector_index_path=_custom_index_path(vi),
            max_features=max_features,
        )
        st.summary = (
            f"{len(shared.schema.columns)} output(s) → "
            f"{len(shared.selected_columns)} selected · 80/10/10 split"
        )
    eq._shared = shared
    knn = None
    if uses_knn:
        with steps(f"Nearest neighbours (k={N_NEIGHBORS})") as st:
            knn = fit_knn(shared=shared, vector_index=vi, k=N_NEIGHBORS)
            st.summary = "self-kNN taken from the index"
    return vi, shared, knn


def _fitters(reference, shared, vi, knn):
    """Zero-argument fitters for each reference score."""
    return {
        "typicality": lambda: Typicality().fit(reference, shared=shared),
        "support": lambda: Support().fit(
            reference, vector_index=vi, k=N_NEIGHBORS, shared=shared, knn=knn
        ),
        "consistency": lambda: Consistency().fit(
            reference, vector_index=vi, k=N_NEIGHBORS, shared=shared, knn=knn
        ),
        "extremity": lambda: Extremity().fit(reference, shared=shared),
        "signal": lambda: Signal().fit(reference, vector_index=vi, shared=shared),
    }


def _validate_reference(reference: pd.DataFrame) -> None:
    """Check the reference table: non-empty, SMILES input column, unique keys."""
    if reference.empty:
        raise SchemaError("Reference DataFrame is empty.")
    validate_input_column(reference)
    check_unique_keys(reference)


def _load_index(
    reference: pd.DataFrame, vector_index: str | pathlib.Path | None
) -> VectorIndex:
    """Load the vector index and check it holds the reference's molecules in order."""
    t = time.perf_counter()
    vi = VectorIndex.load(vector_index or reference_library_path())
    vi.validate_smiles(list(reference["input"]))
    logger.info(
        f"vector index | loaded {vi.library_name or '(unnamed)'} | "
        f"n_ref={vi.n_reference:,} | {time.perf_counter() - t:.1f}s"
    )
    return vi


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
    needs_knn = eq.support is not None or eq.consistency is not None
    steps = console.Steps(1 + needs_knn + len(components))
    with console.section("Reference modality") as section:
        with steps("Validate and scale the query") as st:
            validate_against_schema(query, eq._shared.schema)
            query_repr = _make_query_repr(eq._shared, query)
            st.summary = (
                f"{query_repr.shape[0]:,} molecules · {query_repr.shape[1]} feature(s)"
            )
        neighbours = _shared_neighbours(eq, query, query_repr, steps)
        metadata["n_reference"] = len(eq._shared.reference_ids)
        for name, component in components.items():
            with steps(f"Score: {score_name(name)}") as st:
                result = _run_component(name, component, query, query_repr, neighbours)
                st.summary = console.median_summary(result.score)
            column = score_name(name)
            columns[result.score.name] = result.score
            columns[result.score_raw.name] = result.score_raw
            if name in ("typicality", "extremity"):
                results[name] = result
            if hasattr(result, "score_log"):
                columns[f"{column}_log"] = result.score_log
            metadata.update({f"{column}_{k}": v for k, v in result.metadata.items()})
            logger.info(
                f"score {name!r} | mean={float(result.score.mean()):.4f} "
                f"raw mean={float(result.score_raw.mean()):.4f}"
            )
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


def _run_component(name, component, query, query_repr, neighbours):
    """Run one reference component with the precomputed shared inputs."""
    if name in ("typicality", "extremity"):
        return component.run(query, query_repr=query_repr)
    if name == "support":
        return component.run(
            query,
            query_fp_indices=neighbours["query_fp_indices"],
            query_fp_distances=neighbours["query_fp_distances"],
        )
    if name == "consistency":
        return component.run(query, query_repr=query_repr, **neighbours)
    return component.run(query)


def _shared_neighbours(eq, query: pd.DataFrame, query_repr: np.ndarray, steps) -> dict:
    """FP kNN (and output distances) computed once for Support and Consistency."""
    out = {
        "query_fp_indices": None,
        "query_fp_distances": None,
        "query_output_distances": None,
    }
    if eq.support is None and eq.consistency is None:
        return out
    knn = (eq.support or eq.consistency).knn_
    with steps(f"Nearest library neighbours (k={knn.k})") as st:
        distances, indices = _query_fp_distances(query, eq._get_vector_index(), knn.k)
        out["query_fp_distances"], out["query_fp_indices"] = distances, indices
        if eq.consistency is not None:
            assert eq._shared.ref_repr is not None
            out["query_output_distances"] = _query_output_distances(
                query_repr, eq._shared.ref_repr, indices, distances
            )
        st.summary = f"median nearest similarity {_nanmedian(1 - distances[:, 0]):.2f}"
    return out


def _nanmedian(values: np.ndarray) -> float:
    """Median of the finite values (NaN rows are unparsable SMILES); NaN if none."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


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
            "strings for vector-index alignment."
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
