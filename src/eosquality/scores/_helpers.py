"""Private helpers shared across the score classes.

These functions are used by more than one of Typicality / Support /
Consistency / Extremity (and by the :class:`ErsiliaQuality` orchestrator).
Keeping them in one neutral module avoids the "Support owns
``_query_output_distances`` even though it never uses it" smell.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd

from eosquality.knn.fit import fit_knn
from eosquality.knn.state import KnnFitState
from eosquality.library.identity import reference_library_path
from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import validate_against_schema
from eosquality.shared.fit import fit_shared
from eosquality.shared.state import SharedFitState
from eosquality.vectorindex import VectorIndex


# ---------------------------------------------------------------------------
# Per-component metadata payload
# ---------------------------------------------------------------------------


def _component_metadata(
    *,
    component: str,
    k: int | None,
    fit_timestamp: str | None,
    fit_duration_seconds: float | None,
) -> dict[str, Any]:
    """Build the per-component ``metadata.json`` payload.

    Shared by every score class so the metadata shape stays consistent
    across subfolders. Only component-specific bookkeeping is recorded
    here; shared dataset information (n_samples, n_features,
    eosquality_version) lives once in ``shared/metadata.json``. ``k``
    may be ``None`` for scores that don't use a vector index.
    """
    return {
        "component": component,
        "fit_timestamp": fit_timestamp,
        "fit_duration_seconds": float(fit_duration_seconds or 0.0),
        "k": int(k) if k is not None else None,
    }


# ---------------------------------------------------------------------------
# Shared / kNN state resolution
# ---------------------------------------------------------------------------


def _make_pipeline(shared: SharedFitState) -> PreprocessPipeline:
    """Rebuild a fitted :class:`PreprocessPipeline` from the shared state."""
    return PreprocessPipeline.from_state(
        {
            "schema": shared.schema,
            "scaler_params": shared.scaler_params,
            "binary_class_freq": shared.binary_class_freq,
        }
    )


def _resolve_shared_and_knn(
    *,
    reference: pd.DataFrame,
    vector_index: str | pathlib.Path | VectorIndex,
    k: int,
    eos_id: str | None,
    version: str | None,
    shared: SharedFitState | None,
    knn: KnnFitState | None,
) -> tuple[SharedFitState, KnnFitState, VectorIndex]:
    """Resolve the shared and kNN states, fitting them on demand.

    Parameters
    ----------
    reference:
        Raw reference DataFrame; only consulted if ``shared`` or ``knn``
        is ``None``.
    vector_index:
        Either a path to a VectorIndex folder, or a pre-loaded
        :class:`VectorIndex` instance.
    k, eos_id, version:
        Required only when ``shared`` / ``knn`` need to be fit here.
    shared, knn:
        Optional pre-fit states from a composed orchestrator pass.

    Returns
    -------
    tuple
        ``(shared, knn, loaded_vector_index)``. The third value is the
        actual VectorIndex object (loaded once) so the caller can cache
        it for run-time.
    """
    if isinstance(vector_index, VectorIndex):
        vi = vector_index
    else:
        vi = VectorIndex.load(pathlib.Path(vector_index))

    if shared is None:
        if eos_id is None or version is None:
            raise ValueError(
                "fit needs either a pre-fit shared= argument, or eos_id= "
                "and version= so the shared state can be fit here."
            )
        library_id = str(vi._config.get("library_name", "") or "")
        shared, ref_repr = fit_shared(
            reference, eos_id=eos_id, version=version, library_id=library_id
        )
    else:
        validate_against_schema(reference, shared.schema)
        ref_repr = _make_pipeline(shared).transform(reference)

    if knn is None:
        knn = fit_knn(
            shared=shared,
            ref_repr=ref_repr,
            vector_index=vi,
            k=k,
        )
    return shared, knn, vi


def _resolve_vector_index(shared: SharedFitState) -> VectorIndex:
    """Load the VectorIndex pinned by ``shared.metadata.library_id``.

    Uses :func:`eosquality.library.identity.reference_library_path` so
    saved artifacts stay portable across machines: the canonical
    library is found via env override → repo ``data/indices/`` →
    ``~/.eosquality/`` cache → S3, without baking a path into the
    artifact.
    """
    library_id = shared.metadata.library_id
    if not library_id:
        raise RuntimeError(
            "Cannot resolve a vector index: shared.metadata.library_id is empty. "
            "An index-aware score is loaded but the fit did not tag a library."
        )
    return VectorIndex.load(reference_library_path())


# ---------------------------------------------------------------------------
# Query-time distances (FP and output-space)
# ---------------------------------------------------------------------------


def _query_fp_distances(
    query: pd.DataFrame, vi: VectorIndex, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fp_distances, indices)`` for each query row.

    Wraps :meth:`VectorIndex.query` and returns its Tanimoto-distance
    and index arrays (shape ``(n_query, k)`` each). Used by Support
    at run time and as the entry-point for Consistency's output-space
    derivation.
    """
    query_smiles = list(query["input"])
    fp_distances, vi_indices = vi.query(query_smiles, k=k)
    return fp_distances.astype(np.float64), vi_indices


def _query_output_distances(
    query_repr: np.ndarray, knn: KnnFitState, indices: np.ndarray
) -> np.ndarray:
    """Mean L1 in output space from ``query_repr`` to ``knn.ref_repr[indices]``.

    Takes the FP-selected indices as input — the caller is expected to
    have already obtained them via :func:`_query_fp_distances`. Used by
    Consistency at run time.
    """
    neighbor_reprs = knn.ref_repr[indices]
    diffs = query_repr[:, None, :] - neighbor_reprs
    return np.abs(diffs).sum(axis=2) / query_repr.shape[1]
