"""Private helpers shared across the score classes.

These functions are used by more than one of Typicality / Support /
Consistency / Extremity (and by the :class:`ErsiliaQuality` orchestrator).
Keeping them in one neutral module avoids the "Support owns
``_query_output_distances`` even though it never uses it" smell.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from eosquality.exceptions import IncompatibleArtifactsError
from eosquality.knn.fit import fit_knn
from eosquality.knn.state import KnnFitState
from eosquality.library.identity import LIBRARY_ID, reference_library_path
from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import validate_against_schema
from eosquality.shared.fit import fit_shared
from eosquality.shared.state import SharedFitState
from eosquality.vectorindex import VectorIndex

# ---------------------------------------------------------------------------
# Aggregation + calibration shared by typicality and extremity
# ---------------------------------------------------------------------------


# Quantile used to collapse per-feature values into a single per-row aggregate
# in typicality and extremity. Using Q66 (rather than the mean / Q50) shifts
# the aggregate toward the upper end of the per-feature distribution, so the
# subsequent CDF calibration has more dynamic range at the "typical" /
# "extreme" tail. Both scores share one source of truth here.
AGGREGATE_QUANTILE = 0.66


def _cdf_score(
    values: np.ndarray,
    sorted_self: np.ndarray,
    n_reference: int,
    *,
    higher_is_higher: bool,
) -> np.ndarray:
    """Map per-row values to calibrated scores via the reference CDF.

    Single source of truth for every CDF-calibrated score in the
    package. ``higher_is_higher=True`` (typicality, extremity, signal):
    a value above the reference median maps to a score above 0.5.
    ``higher_is_higher=False`` (support, consistency — distance-based):
    a *smaller* value (closer to the reference) maps to a score above
    0.5 via a ``1 − cdf`` flip.

    Formula: ``cdf = searchsorted(sorted_self, values, side='right') /
    n_reference``; return ``clip(cdf if higher_is_higher else 1 - cdf,
    eps, 1.0)`` with ``eps = 1 / (2 · n_reference)``.
    """
    eps = 1.0 / (2.0 * max(n_reference, 1))
    cdf = np.searchsorted(sorted_self, values, side="right") / n_reference
    out = cdf if higher_is_higher else 1.0 - cdf
    return np.clip(out, eps, 1.0)


def _score_from_aggregates(
    aggregates: np.ndarray,
    sorted_self_aggregates: np.ndarray,
    n_reference: int,
) -> np.ndarray:
    """``higher_is_higher=True`` wrapper around :func:`_cdf_score`.

    Used by typicality / extremity / signal where the per-row aggregate
    grows with the property being measured (more typical, more extreme,
    more signal) and a query above the reference median should score
    above 0.5.
    """
    return _cdf_score(
        aggregates,
        sorted_self_aggregates,
        n_reference,
        higher_is_higher=True,
    )


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


def _make_query_repr(shared: SharedFitState, df: pd.DataFrame) -> np.ndarray:
    """Scale ``df`` via the shared pipeline and project to the selected features.

    Every score that needs a per-row representation in output space should
    use this helper instead of calling the pipeline directly, so feature
    selection is applied consistently at fit time and run time.
    """
    return shared.filter_features(_make_pipeline(shared).transform(df))


def _reference_repr(shared: SharedFitState, reference: pd.DataFrame) -> np.ndarray:
    """Scaled + feature-selected reference matrix for fit-time use.

    Reuses ``shared.ref_repr`` (computed once by :func:`fit_shared`) when it
    is row-aligned with ``reference``; otherwise re-applies the pipeline.
    Avoids re-running the eosframes transform over the full reference in
    every score's ``fit``.
    """
    ref_repr = shared.ref_repr
    if (
        ref_repr is not None
        and ref_repr.shape[0] == len(reference)
        and list(reference.index) == list(shared.reference_ids)
    ):
        return ref_repr
    return _make_query_repr(shared, reference)


def _resolve_shared(
    reference: pd.DataFrame,
    *,
    shared: SharedFitState | None,
    eos_id: str | None,
    version: str | None,
    component: str,
) -> SharedFitState:
    """Return ``shared`` (validated against ``reference``) or fit it here."""
    if shared is not None:
        validate_against_schema(reference, shared.schema)
        return shared
    if eos_id is None or version is None:
        raise ValueError(
            f"{component}.fit needs either a pre-fit shared= argument, or "
            "eos_id= and version= so it can fit the shared state itself."
        )
    return fit_shared(reference, eos_id=eos_id, version=version)


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
        shared = fit_shared(
            reference,
            eos_id=eos_id,
            version=version,
            library_id=vi.library_name,
            vector_index_path=_custom_index_path(vi),
        )
    else:
        validate_against_schema(reference, shared.schema)

    if knn is None:
        knn = fit_knn(shared=shared, vector_index=vi, k=k)
    return shared, knn, vi


def _custom_index_path(vi: VectorIndex) -> str:
    """Absolute index folder for a non-canonical index, ``""`` for the canonical one."""
    if vi.library_name == LIBRARY_ID:
        return ""
    return str(vi.index_dir.resolve())


def _resolve_vector_index(shared: SharedFitState) -> VectorIndex:
    """Load the VectorIndex the artifact was fit against.

    Artifacts fit on the canonical library (``library_id == LIBRARY_ID``)
    resolve it via :func:`eosquality.library.identity.reference_library_path`
    (env override → ``./data/indices/`` → ``~/.eosquality/`` cache), so they
    stay portable across machines. Artifacts fit on a custom index record
    its absolute folder in ``shared.metadata.vector_index_path`` and load
    it from there. Either way the loaded index's ``library_name`` must equal
    the ``library_id`` recorded at fit time.
    """
    library_id = shared.metadata.library_id
    if not library_id:
        raise RuntimeError(
            "Cannot resolve a vector index: shared.metadata.library_id is empty. "
            "An index-aware score is loaded but the fit did not tag a library."
        )
    if library_id == LIBRARY_ID:
        path = reference_library_path()
    elif shared.metadata.vector_index_path:
        path = pathlib.Path(shared.metadata.vector_index_path)
    else:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit against reference library {library_id!r} but "
            f"this install ships {LIBRARY_ID!r}. Install a compatible "
            "eosquality release or refit against the current library."
        )
    vi = VectorIndex.load(path)
    if vi.library_name != library_id:
        raise IncompatibleArtifactsError(
            f"Vector index at {path} is library {vi.library_name!r}, but the "
            f"artifacts were fit against {library_id!r}."
        )
    return vi


# ---------------------------------------------------------------------------
# Query-time distances (FP and output-space)
# ---------------------------------------------------------------------------


# Tanimoto distance below this is treated as a perfect-fingerprint match
# (i.e. the query molecule is present in the library, possibly under a
# different canonical SMILES). FPSim2 returns 0.0 in that case; the small
# epsilon guards against float-precision wobble.
_SELF_MATCH_DISTANCE_THRESHOLD = 1e-6


def _query_fp_distances(
    query: pd.DataFrame,
    vi: VectorIndex,
    k: int,
    *,
    exclude_self_match: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fp_distances, indices)`` for each query row, shape ``(n_query, k)``.

    Wraps :meth:`VectorIndex.query`. The reference's calibration CDF is
    built from **identity-stripped** self-kNN distances (each library
    row's k nearest neighbors are k *other* molecules — see the index
    build step). For the per-query lookup to be comparable, we strip
    perfect-Tanimoto self-matches here too: queries that are already in
    the library would otherwise return themselves as the top neighbor
    with distance 0, inflating their score artificially.

    To do this we query ``k + 1`` neighbors and drop one column per row:
    the first perfect-match entry if any exists, else the furthest of
    the ``k + 1``. Pass ``exclude_self_match=False`` to opt out (useful
    if you've already pre-stripped the query SMILES, or for direct
    raw-distance debugging).
    """
    query_smiles = list(query["input"])
    if not exclude_self_match:
        fp_distances, vi_indices = vi.query(query_smiles, k=k)
        return fp_distances.astype(np.float64), vi_indices

    fp_distances, vi_indices = vi.query(query_smiles, k=k + 1)
    fp_distances = fp_distances.astype(np.float64)
    n_query = fp_distances.shape[0]

    zero_mask = fp_distances < _SELF_MATCH_DISTANCE_THRESHOLD  # (n_query, k+1)
    has_zero = zero_mask.any(axis=1)
    first_zero = zero_mask.argmax(axis=1)  # 0 when has_zero is False (unused)
    # Column to drop per row: first perfect match if any, else the furthest
    # (the (k+1)-th entry, index k).
    drop_col = np.where(has_zero, first_zero, k)

    all_cols = np.arange(k + 1)
    keep_mask = all_cols[None, :] != drop_col[:, None]  # (n_query, k+1)

    fp_kept = fp_distances[keep_mask].reshape(n_query, k)
    idx_kept = vi_indices[keep_mask].reshape(n_query, k)
    return fp_kept, idx_kept


# Rows per chunk when computing output-space neighbor distances. Bounds the
# ``(chunk, k, n_features)`` temporary instead of materialising it for the
# whole reference (~0.5 GB at 1.35M × 5 × 10 in float64).
_OUTPUT_DISTANCE_CHUNK = 65_536


def _query_output_distances(
    query_repr: np.ndarray, ref_repr: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Mean L1 in output space from ``query_repr`` to ``ref_repr[indices]``.

    Returns ``(n_query, k)``: for each query row and each of its k
    FP-selected neighbors, the mean absolute difference over features.
    ``indices`` come from :func:`_query_fp_distances` (run time) or the
    precomputed self-kNN (fit time). ``ref_repr`` is the post-reduction
    scaled reference matrix, ``SharedFitState.ref_repr``. Computed in
    row chunks to keep peak memory flat for reference-sized inputs.
    """
    n_query, n_features = query_repr.shape
    out = np.empty(indices.shape, dtype=np.float64)
    for start in range(0, n_query, _OUTPUT_DISTANCE_CHUNK):
        stop = min(start + _OUTPUT_DISTANCE_CHUNK, n_query)
        diffs = query_repr[start:stop, None, :] - ref_repr[indices[start:stop]]
        np.abs(diffs, out=diffs)
        out[start:stop] = diffs.sum(axis=2) / n_features
    return out
