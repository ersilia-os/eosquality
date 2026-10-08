"""Private helpers shared across the score classes.

These functions are used by more than one score (and by the
:class:`ErsiliaQuality` orchestrator). Keeping them in one neutral module
avoids one score owning a helper that others also use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import validate_against_schema
from eosquality.shared.fit import fit_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.parallel import map_rows

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


# Rows per block in _row_nanquantile.
_QUANTILE_CHUNK = 262_144


def _aggregate_percentiles(per_feature_pct: np.ndarray) -> np.ndarray:
    """Per-row Q66 of the per-feature percentiles (NaN where none is finite)."""
    if per_feature_pct.shape[1] == 0:
        return np.full(per_feature_pct.shape[0], np.nan)
    return _row_nanquantile(per_feature_pct, AGGREGATE_QUANTILE)


def _row_nanquantile(values: np.ndarray, q: float) -> np.ndarray:
    """Row-wise ``np.nanquantile(values, q, axis=1)`` (linear method), vectorised.

    numpy's ``nanquantile`` along an axis falls back to a Python loop over
    rows, about 100 s for the 1.35M-row reference library. Sorting each row
    (NaN last) and interpolating at ``q * (n_finite - 1)`` gives identical
    values in well under a second. All-NaN rows (and zero columns) give NaN.
    It differs from numpy only for infinite values, which numpy turns into
    NaN when interpolating; no caller produces them.

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
    values = np.asarray(values)
    n_rows = values.shape[0]
    if values.ndim != 2 or values.shape[1] == 0:
        return np.full(n_rows, np.nan)
    # Row blocks bound the temporary copies (sorted values, NaN mask) to a few
    # hundred MB, whatever the number of rows.
    out = np.empty(n_rows, dtype=np.float64)
    for start in range(0, n_rows, _QUANTILE_CHUNK):
        stop = min(start + _QUANTILE_CHUNK, n_rows)
        out[start:stop] = _row_nanquantile_block(values[start:stop], q)
    return out


def _row_nanquantile_block(values: np.ndarray, q: float) -> np.ndarray:
    """:func:`_row_nanquantile` for one block of rows."""
    values = values.astype(np.float64, copy=False)
    n_rows = values.shape[0]
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

    - ``higher_is_higher=True`` (typicality, extremity): a value
      above the reference median maps above 0.5.
    - ``higher_is_higher=False`` (distances): a
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

    Used by typicality / extremity, where the per-row aggregate
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
# Shared state resolution
# ---------------------------------------------------------------------------


def _make_pipeline(shared: SharedFitState) -> PreprocessPipeline:
    """Rebuild a fitted :class:`PreprocessPipeline` from the shared state."""
    return PreprocessPipeline.from_state(
        {
            "schema": shared.schema,
            "scaler_params": shared.scaler_params,
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


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

# Below this many molecules a process pool costs more than it saves (each worker
# imports RDKit); ``standardize_all`` spreads larger lists inside ``parallel.workers``.
_STANDARDIZE_MIN_PARALLEL = 20_000


def _standardize(smiles: str) -> str | None:
    """Largest fragment, then RDKit canonical isomeric SMILES; ``None`` if unparsable.

    Used to match training molecules with queries: salt and solvent forms of
    the same parent molecule map to one standardised SMILES.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    with rdBase.BlockLogs():  # RDKit's sanitisation notes are not actionable here
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
        return Chem.MolToSmiles(mol)


def _standardize_all(smiles: Sequence) -> list[str | None]:
    """:func:`_standardize` of every SMILES, in order (parallel inside ``workers``).

    Parameters
    ----------
    smiles : sequence
        SMILES strings (anything else standardises to ``None``).

    Returns
    -------
    list of str or None
    """
    out = np.empty(len(smiles), dtype=object)
    map_rows(
        _standardize,
        smiles,
        out,
        label="standardise",
        min_items=_STANDARDIZE_MIN_PARALLEL,
    )
    return out.tolist()
