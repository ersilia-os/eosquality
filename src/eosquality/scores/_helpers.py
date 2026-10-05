"""Private helpers shared across the score classes.

These functions are used by more than one of Typicality / Support /
Consistency / Extremity (and by the :class:`ErsiliaQuality` orchestrator).
Keeping them in one neutral module avoids the "Support owns
``_query_output_distances`` even though it never uses it" smell.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from eosquality.exceptions import IncompatibleArtifactsError
from eosquality.knn.fit import fit_knn
from eosquality.knn.state import KnnFitState
from eosquality.library.identity import LIBRARY_ID, reference_library_path
from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import validate_against_schema
from eosquality.shared.fit import fit_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger
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


def _nan_aggregate(per_feature: np.ndarray) -> np.ndarray:
    """Per-row ``AGGREGATE_QUANTILE`` over the finite features of each row.

    NaN features carry no information and are ignored; a row whose every
    feature is NaN aggregates to NaN. Shared by typicality and extremity so
    both follow the same missing-value policy.
    """
    return _row_nanquantile(per_feature, AGGREGATE_QUANTILE)


def _row_nanquantile(values: np.ndarray, q: float) -> np.ndarray:
    """Row-wise ``np.nanquantile(values, q, axis=1)`` (linear method), vectorised.

    numpy's ``nanquantile`` along an axis falls back to a Python loop over
    rows, about 100 s for the 1.35M-row reference library. Sorting each row
    (NaN last) and interpolating at ``q * (n_finite - 1)`` gives identical
    values in well under a second. All-NaN rows (and zero columns) give NaN.

    Parameters
    ----------
    values : numpy.ndarray
        ``(n_rows, n_columns)`` array; NaN entries are ignored.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``(n_rows,)`` float64.
    """
    values = np.asarray(values, dtype=np.float64)
    n_rows = values.shape[0]
    if values.shape[1] == 0:
        return np.full(n_rows, np.nan)
    ordered = np.sort(values, axis=1)  # NaN sorts last
    n_finite = np.count_nonzero(~np.isnan(values), axis=1)
    has_values = n_finite > 0
    position = q * np.maximum(n_finite - 1, 0)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, np.maximum(n_finite - 1, 0))
    fraction = position - lower
    rows = np.arange(n_rows)
    a = ordered[rows, lower]
    b = ordered[rows, upper]
    # numpy's _lerp: interpolate from whichever end is closer, for exactness.
    diff = b - a
    out = np.where(fraction >= 0.5, b - diff * (1.0 - fraction), a + diff * fraction)
    out = np.where(fraction == 0, a, out)
    return np.where(has_values, out, np.nan)


def _cdf_score(
    values: np.ndarray,
    sorted_self: np.ndarray,
    *,
    higher_is_higher: bool,
) -> np.ndarray:
    """Map per-row values to calibrated scores via the reference CDF.

    Single source of truth for every CDF-calibrated score in the package.
    ``sorted_self`` is the ascending array of the same raw quantity computed
    on the reference (finite values only).

    - ``higher_is_higher=True`` (typicality, extremity, signal): a value
      above the reference median maps above 0.5.
    - ``higher_is_higher=False`` (support, consistency — distances): a
      *smaller* value maps above 0.5 via a ``1 − cdf`` flip.

    The CDF uses **mid-ranks**, ``cdf = (#{ref < v} + #{ref ≤ v}) / (2n)``,
    so a value tied with many reference rows sits in the middle of its tie
    block instead of at its top. Reference rows scored against their own
    CDF therefore average exactly 0.5 even for heavily tied raw values
    (e.g. a one-output model, or quantised typicality). Results are
    clipped to ``[eps, 1]`` with ``eps = 1 / (2n)``; NaN values stay NaN.
    """
    n = sorted_self.size
    if n == 0:
        raise ValueError("Cannot calibrate against an empty reference distribution.")
    values = np.asarray(values, dtype=np.float64)
    below = np.searchsorted(sorted_self, values, side="left")
    at_or_below = np.searchsorted(sorted_self, values, side="right")
    cdf = (below + at_or_below) / (2.0 * n)
    out = np.clip(cdf if higher_is_higher else 1.0 - cdf, 1.0 / (2.0 * n), 1.0)
    return np.where(np.isnan(values), np.nan, out)


def _score_from_aggregates(
    aggregates: np.ndarray, sorted_self_aggregates: np.ndarray
) -> np.ndarray:
    """``higher_is_higher=True`` wrapper around :func:`_cdf_score`.

    Used by typicality / extremity / signal, where the per-row aggregate
    grows with the property being measured.
    """
    return _cdf_score(aggregates, sorted_self_aggregates, higher_is_higher=True)


def _sorted_finite(values: np.ndarray, component: str) -> np.ndarray:
    """Ascending float64 copy of the finite entries of ``values`` (the CDF table)."""
    finite = np.sort(values[np.isfinite(values)]).astype(np.float64)
    if finite.size == 0:
        raise ValueError(
            f"{component}.fit: no reference row has a finite raw value, so no "
            "CDF calibration is possible. Check the reference data."
        )
    return finite


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


# Tanimoto distance below this is a perfect fingerprint match. FPSim2 returns
# exactly 0.0; the epsilon guards against float wobble.
_SELF_MATCH_DISTANCE_THRESHOLD = 1e-6


def _canonical(smiles: str) -> str | None:
    """RDKit canonical isomeric SMILES, or ``None`` if it does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _standardize(smiles: str) -> str | None:
    """Largest fragment, then RDKit canonical isomeric SMILES; ``None`` if unparsable.

    Used to match training molecules with queries: salt and solvent forms of
    the same parent molecule map to one standardised SMILES.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    return Chem.MolToSmiles(mol)


def _is_same_molecule(query_smiles: str, library_smiles: str) -> bool:
    if query_smiles == library_smiles:
        return True
    a, b = _canonical(query_smiles), _canonical(library_smiles)
    return a is not None and a == b


def _query_fp_distances(
    query: pd.DataFrame,
    vi: VectorIndex,
    k: int,
    *,
    exclude_self_match: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fp_distances, indices)`` for each query row, shape ``(n_query, k)``.

    Wraps :meth:`VectorIndex.query`. The reference's calibration CDFs are
    built from **identity-stripped** self-kNN (each library row's k nearest
    neighbors are k *other* molecules). For queries to be comparable, a
    query that *is* a library molecule must not count itself as a
    neighbor: we query ``k + 1`` neighbors and drop the zero-distance
    neighbor whose library SMILES is the query molecule (string match, or
    same RDKit canonical isomeric SMILES). Rows without such a match drop
    their furthest neighbor instead. A zero-distance neighbor that is a
    *different* molecule (e.g. a stereoisomer sharing the Morgan
    fingerprint) is kept, exactly as in the library's own self-kNN.

    Pass ``exclude_self_match=False`` to return the raw top-k.

    Rows whose SMILES is missing or does not parse get NaN distances (and
    index 0, a placeholder), so structure-based scores are NaN for them
    instead of the whole batch failing inside FPSim2; a warning names them.
    """
    query_smiles = list(query["input"])
    valid = np.array([_parses(s) for s in query_smiles], dtype=bool)
    if valid.all():
        return _query_fp_distances_valid(query_smiles, vi, k, exclude_self_match)
    bad = np.flatnonzero(~valid)
    shown = ", ".join(str(query.index[i]) for i in bad[:5])
    logger.warning(
        f"{len(bad):,} query row(s) have a missing or unparsable SMILES "
        f"(rows {shown}{', …' if len(bad) > 5 else ''}); their structure-based "
        "scores are NaN."
    )
    distances = np.full((len(query_smiles), k), np.nan)
    indices = np.zeros((len(query_smiles), k), dtype=np.int64)
    if valid.any():
        d, i = _query_fp_distances_valid(
            [query_smiles[j] for j in np.flatnonzero(valid)], vi, k, exclude_self_match
        )
        distances[valid], indices[valid] = d, i
    return distances, indices


def _parses(smiles) -> bool:
    """Whether ``smiles`` is a non-empty string that RDKit can parse."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles) is not None


def _query_fp_distances_valid(
    query_smiles: list[str], vi: VectorIndex, k: int, exclude_self_match: bool
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_query_fp_distances` for SMILES that are known to parse."""
    if not exclude_self_match:
        fp_distances, vi_indices = vi.query(query_smiles, k=k)
        return fp_distances.astype(np.float64), vi_indices

    fp_distances, vi_indices = vi.query(query_smiles, k=k + 1)
    fp_distances = fp_distances.astype(np.float64)
    n_query = fp_distances.shape[0]
    library_smiles = vi.smiles

    # Column to drop per row: the self match if present, else the furthest.
    drop_col = np.full(n_query, k, dtype=np.int64)
    rows, cols = np.nonzero(fp_distances < _SELF_MATCH_DISTANCE_THRESHOLD)
    for i, j in zip(rows, cols, strict=True):
        if drop_col[i] != k:
            continue  # already found this row's self match
        if _is_same_molecule(query_smiles[i], library_smiles[vi_indices[i, j]]):
            drop_col[i] = j

    keep_mask = np.arange(k + 1)[None, :] != drop_col[:, None]
    fp_kept = fp_distances[keep_mask].reshape(n_query, k)
    idx_kept = vi_indices[keep_mask].reshape(n_query, k)
    return fp_kept, idx_kept


# Rows per chunk when computing output-space neighbor distances. Bounds the
# ``(chunk, k, n_features)`` temporary instead of materialising it for the
# whole reference (~0.5 GB at 1.35M × 5 × 10 in float64).
_OUTPUT_DISTANCE_CHUNK = 65_536


def _query_output_distances(
    query_repr: np.ndarray,
    ref_repr: np.ndarray,
    indices: np.ndarray,
    fp_distances: np.ndarray | None = None,
) -> np.ndarray:
    """Mean L1 in output space from ``query_repr`` to ``ref_repr[indices]``.

    Returns ``(n_query, k)``: for each query row and each of its k
    FP-selected neighbors, the mean absolute difference over the features
    that are finite on both sides (NaN if none are).
    ``indices`` come from :func:`_query_fp_distances` (run time) or the
    precomputed self-kNN (fit time). ``ref_repr`` is the post-reduction
    scaled reference matrix, ``SharedFitState.ref_repr``. Computed in
    row chunks to keep peak memory flat for reference-sized inputs. Pass the
    matching ``fp_distances`` to get NaN wherever a neighbour is only a
    placeholder (unparsable query SMILES, see :func:`_query_fp_distances`).
    """
    n_query = query_repr.shape[0]
    out = np.empty(indices.shape, dtype=np.float64)
    for start in range(0, n_query, _OUTPUT_DISTANCE_CHUNK):
        stop = min(start + _OUTPUT_DISTANCE_CHUNK, n_query)
        diffs = query_repr[start:stop, None, :] - ref_repr[indices[start:stop]]
        np.abs(diffs, out=diffs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN
            out[start:stop] = np.nanmean(diffs, axis=2)
    if fp_distances is not None:
        out[~np.isfinite(fp_distances)] = np.nan  # placeholder neighbours
    return out
