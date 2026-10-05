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
    ALL_SCORES,
    INDEX_AWARE,
    KNN_USERS,
    MIN_REFERENCE_SAMPLES,
    SCORE_ORDER,
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
    ignore_size: bool,
    scores: Iterable[str],
    max_features: int | None,
    max_signal_train_samples: int | None,
    signal_descriptor: str,
) -> None:
    """Fit the reference-modality components of ``eq`` on the reference predictions.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator to fill (``_shared``, the score attributes).
    reference : pandas.DataFrame
        Predictions on the reference library.
    eos_id, version : str
        Model identifier and dataset version.
    vector_index : str, pathlib.Path or None
        Custom index folder, or ``None`` for the canonical library.
    ignore_size : bool
        Skip the minimum-size check.
    scores : iterable of str
        Reference scores to fit.
    max_features : int or None
        Feature-selection cap.
    max_signal_train_samples : int or None
        Signal training rows.
    signal_descriptor : str
        Signal descriptor backend.
    """
    scores_set = set(scores)
    requested = [n for n in SCORE_ORDER if n in scores_set]
    uses_index = bool(scores_set & INDEX_AWARE)
    uses_knn = bool(scores_set & KNN_USERS)
    steps = console.Steps(2 + uses_index + uses_knn + len(requested))
    with console.section("Reference modality") as section:
        with steps("Validate reference predictions") as st:
            _validate_reference(reference, scores, ignore_size)
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
            vector_index=vector_index if uses_index else False,
            max_features=max_features,
            uses_knn=uses_knn,
        )
        k = eq.config.neighbors.k
        fitters = _fitters(
            reference, shared, vi, knn, k, signal_descriptor, max_signal_train_samples
        )
        for name in requested:
            with steps(f"Score: {name}"):
                setattr(eq, name, fitters[name]())
                anchor = getattr(getattr(eq, name), f"reference_{name}_", None)
                logger.info(f"score {name!r} | fitted | reference={anchor}")
        section.summary = f"{len(requested)} score(s) fitted"
    eq._vector_index_cache = vi


def _fit_upstream(
    eq, reference, steps, *, eos_id, version, vector_index, max_features, uses_knn
):
    """Index, shared state and kNN steps; return ``(vi, shared, knn)``.

    ``vector_index=False`` means no score needs the index.
    """
    vi = None
    if vector_index is not False:
        with steps("Load the vector index") as st:
            vi = _load_index(reference, vector_index)
            st.summary = (
                f"{vi.library_name or 'custom index'} · "
                f"{vi.n_reference:,} molecules match the reference"
            )
    with steps("Shared state: schema, scaling, feature selection, splits") as st:
        shared = fit_shared(
            reference,
            eos_id=eos_id,
            version=version,
            library_id=vi.library_name if vi is not None else "",
            vector_index_path=_custom_index_path(vi) if vi is not None else "",
            max_features=max_features,
        )
        st.summary = (
            f"{len(shared.schema.columns)} output(s) → "
            f"{len(shared.selected_columns)} selected · 80/10/10 split"
        )
    eq._shared = shared
    knn = None
    if uses_knn:
        k = eq.config.neighbors.k
        with steps(f"Nearest neighbours (k={k})") as st:
            knn = fit_knn(shared=shared, vector_index=vi, k=k)
            st.summary = "self-kNN taken from the index"
    return vi, shared, knn


def _fitters(reference, shared, vi, knn, k, signal_descriptor, max_signal_samples):
    """Zero-argument fitters for each reference score."""
    return {
        "typicality": lambda: Typicality().fit(reference, shared=shared),
        "support": lambda: Support().fit(
            reference, vector_index=vi, k=k, shared=shared, knn=knn
        ),
        "consistency": lambda: Consistency().fit(
            reference, vector_index=vi, k=k, shared=shared, knn=knn
        ),
        "extremity": lambda: Extremity().fit(reference, shared=shared),
        "signal": lambda: Signal().fit(
            reference,
            vector_index=vi,
            shared=shared,
            descriptor=signal_descriptor,
            max_train_samples=max_signal_samples,
        ),
    }


def _validate_reference(
    reference: pd.DataFrame, scores: Iterable[str], ignore_size: bool
) -> set[str]:
    """Check the requested scores and the reference table; return the score set."""
    scores_set = set(scores)
    unknown = scores_set - set(ALL_SCORES)
    if unknown:
        raise ValueError(
            f"Unknown score(s): {sorted(unknown)}. Valid choices: {ALL_SCORES}."
        )
    if reference.empty:
        raise SchemaError("Reference DataFrame is empty.")
    if scores_set & INDEX_AWARE:
        validate_input_column(reference)
    if not ignore_size and len(reference) < MIN_REFERENCE_SAMPLES:
        raise ValueError(
            f"Reference dataset has {len(reference):,} rows. Fitting requires "
            f"at least {MIN_REFERENCE_SAMPLES:,} rows for reliable results. "
            "Pass ignore_size=True to bypass this check (not recommended "
            "for production use)."
        )
    check_unique_keys(reference)
    return scores_set


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
) -> None:
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
    """
    assert eq._shared is not None
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
            with steps(f"Score: {name}") as st:
                result = _run_component(name, component, query, query_repr, neighbours)
                st.summary = f"median {float(result.score.median()):.3f}"
            columns[name] = result.score
            columns[f"{name}_raw"] = result.score_raw
            if hasattr(result, "score_log"):
                columns[f"{name}_log"] = result.score_log
            metadata.update({f"{name}_{k}": v for k, v in result.metadata.items()})
            logger.info(
                f"score {name!r} | mean={float(result.score.mean()):.4f} "
                f"raw mean={float(result.score_raw.mean()):.4f}"
            )
        section.summary = f"{len(components)} score(s)"


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
                query_repr, eq._shared.ref_repr, indices
            )
        st.summary = (
            f"median nearest similarity {float(np.median(1 - distances[:, 0])):.2f}"
        )
    return out


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
