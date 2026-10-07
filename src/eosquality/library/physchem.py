"""Physchem descriptor matrix for the reference library.

At library-build time, compute the full set of RDKit physicochemical
descriptors (``rdkit.Chem.Descriptors._descList``, ~200 descriptors)
for every molecule and persist two artifacts alongside the Morgan
fingerprint files:

- ``physchem_scaled.npy`` — ``(n_ref, n_desc)`` float16; non-finite
  RDKit values are replaced by the per-column median, then
  standard-scaled. float16 keeps the file small (~half the size of
  float32) — well within the precision useful for standardized
  values that mostly sit in ``[-3, 3]``.
- ``physchem_scaler.json`` — bundles both the imputer parameters
  (``median`` per descriptor) and the StandardScaler parameters
  (``mean``, ``scale`` per descriptor), plus ``descriptor_names`` and
  the RDKit / scikit-learn versions used. Everything needed to apply
  the identical impute-then-standardize transform to a new molecule
  at run time lives here.

The same :func:`compute_physchem_raw` + :func:`apply_scaler` pair computes
query rows at run time, so reference and query descriptors match exactly.
Large inputs (library builds) are computed with a process pool; small ones
(run-time queries) in-process.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import sklearn
from rdkit import Chem, rdBase
from rdkit import __version__ as _RDKIT_VERSION
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

from eosquality.utils.logging import logger
from eosquality.utils.parallel import map_rows

PHYSCHEM_SCALED_FILE = "physchem_scaled.npy"
PHYSCHEM_SCALER_FILE = "physchem_scaler.json"


# RDKit's canonical descriptor list — (name, callable) tuples. Captured
# at import time so every worker (under multiprocessing 'spawn') sees
# the same ordering after re-importing this module.
DESCRIPTOR_FNS: list[tuple[str, callable]] = list(Descriptors._descList)
DESCRIPTOR_NAMES: list[str] = [name for name, _ in DESCRIPTOR_FNS]
N_DESCRIPTORS: int = len(DESCRIPTOR_FNS)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


_F32_MAX = float(np.finfo(np.float32).max)


def _compute_one(smi: str) -> np.ndarray:
    """Compute all RDKit descriptors for one SMILES.

    Returns a ``(N_DESCRIPTORS,)`` float32 row. On any failure (parse
    error, descriptor exception) the entire row is filled with
    ``np.nan`` so downstream median-imputation handles it uniformly.
    Module-level so it is picklable for ``multiprocessing.Pool``.

    Some RDKit descriptors (``Ipc`` in particular) can blow past
    float32's ~3.4e38 range on pathological inputs; those values are
    coerced to NaN before assignment rather than overflowing to ``±inf``
    (and emitting a numpy RuntimeWarning).
    """
    row = np.full(N_DESCRIPTORS, np.nan, dtype=np.float32)
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smi)
    except Exception:
        return row
    if mol is None:
        return row
    for i, (_, fn) in enumerate(DESCRIPTOR_FNS):
        try:
            value = float(fn(mol))
        except Exception:
            continue  # row[i] stays NaN from the initial fill
        if not np.isfinite(value) or abs(value) > _F32_MAX:
            continue  # NaN-impute downstream rather than store ±inf
        row[i] = value
    return row


# ---------------------------------------------------------------------------
# Raw matrix
# ---------------------------------------------------------------------------


def compute_physchem_raw(
    smiles: Iterable[str],
    *,
    n_jobs: int | None = None,
    show_progress: bool | None = None,
    label: str = "physchem descriptors",
) -> np.ndarray:
    """Compute the ``(n, N_DESCRIPTORS)`` raw float32 descriptor matrix.

    Rows are in input order. Raw values may include NaN; imputation happens
    in :func:`fit_scaler` and :func:`apply_scaler`, not here.

    Parameters
    ----------
    smiles : iterable of str
        Input SMILES.
    n_jobs : int, optional
        Worker processes (default: in-process; ``-1``: every CPU).
    show_progress : bool, optional
        Show a progress bar; ``None`` shows it only for parallel runs.
    label : str, optional
        Progress-bar title.

    Returns
    -------
    numpy.ndarray
        ``(n, N_DESCRIPTORS)`` float32 raw descriptors; unparsable SMILES give a NaN row.
    """
    smiles_list = list(smiles)
    out = np.empty((len(smiles_list), N_DESCRIPTORS), dtype=np.float32)
    return map_rows(
        _compute_one,
        smiles_list,
        out,
        label=label,
        n_jobs=n_jobs,
        chunksize=256,
        show_progress=show_progress,
    )


# ---------------------------------------------------------------------------
# Scaler (imputer + StandardScaler) — fit & apply
# ---------------------------------------------------------------------------


def fit_scaler(raw: np.ndarray) -> dict:
    """Fit the per-column median imputer + StandardScaler.

    Non-finite entries (``NaN``, ``±inf``) are replaced with the
    per-column median (computed over finite values). StandardScaler is
    then fit on the imputed matrix. Returns a JSON-serialisable dict
    with every parameter needed by :func:`apply_scaler`.

    Parameters
    ----------
    raw : numpy.ndarray
        ``(n, N_DESCRIPTORS)`` raw descriptor matrix.

    Returns
    -------
    dict
        ``descriptor_names``, ``median``, ``mean``, ``scale`` and versions.
    """
    if raw.ndim != 2 or raw.shape[1] != N_DESCRIPTORS:
        raise ValueError(f"fit_scaler expected (n, {N_DESCRIPTORS}); got {raw.shape}.")
    finite = np.where(np.isfinite(raw), raw.astype(np.float64), np.nan)
    median = np.nanmedian(finite, axis=0)
    # Columns that are entirely non-finite have median=NaN — fall back to 0
    # so apply_scaler is well-defined.
    all_nan = np.isnan(median)
    if all_nan.any():
        bad = [DESCRIPTOR_NAMES[i] for i in np.flatnonzero(all_nan)]
        logger.warning(
            f"physchem | {int(all_nan.sum())} descriptor(s) had no finite "
            f"values across the reference; imputing with 0 → {bad}"
        )
        median = np.where(all_nan, 0.0, median)

    imputed = np.where(np.isfinite(raw), raw.astype(np.float64), median[None, :])
    scaler = StandardScaler().fit(imputed)
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)

    return {
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "median": median.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "rdkit_version": _RDKIT_VERSION,
        "sklearn_version": sklearn.__version__,
    }


def canonical_scaler() -> dict:
    """The reference library's physchem scaler, shipped with the package.

    ``physchem_scaler.json`` next to this module is a copy of the file
    ``eosquality build`` wrote for the canonical library (impute medians,
    means and scales over its 1.35M molecules). It lets the training modality
    place every model in the same descriptor space without the library
    installed.

    Returns
    -------
    dict
        ``descriptor_names``, ``median``, ``mean``, ``scale`` and the versions
        it was fitted with.

    Raises
    ------
    RuntimeError
        If the installed RDKit's descriptor list differs from the scaler's.
    """
    import json
    from importlib import resources

    with (
        resources.files("eosquality.library").joinpath(PHYSCHEM_SCALER_FILE).open() as f
    ):
        params = json.load(f)
    check_descriptor_names(params)
    return params


def check_descriptor_names(scaler_params: dict) -> None:
    """Raise if the installed RDKit's descriptor list differs from the fitted one.

    Query descriptors are computed column-by-column from
    ``Descriptors._descList``; if RDKit added, removed or reordered a
    descriptor since the library was built, columns would silently
    misalign with the scaler and the trained model.

    Parameters
    ----------
    scaler_params : dict
        Parameters from :func:`fit_scaler` (``descriptor_names`` is checked).
    """
    fitted = list(scaler_params.get("descriptor_names", []))
    if fitted != DESCRIPTOR_NAMES:
        raise RuntimeError(
            "RDKit physchem descriptor list mismatch: the library was built "
            f"with {len(fitted)} descriptors (RDKit "
            f"{scaler_params.get('rdkit_version', '?')}), this environment has "
            f"{N_DESCRIPTORS} (RDKit {_RDKIT_VERSION}). Install the RDKit "
            "version the library was built with."
        )


def apply_scaler(raw: np.ndarray, scaler_params: dict) -> np.ndarray:
    """Apply impute → standard-scale using persisted parameters.

    Mirrors :func:`fit_scaler`: non-finite entries replaced by
    ``scaler_params["median"]``, then ``(x - mean) / scale``. Columns
    whose ``scale`` is 0 (constant in the reference) are divided by 1
    to avoid division-by-zero, matching scikit-learn's internal
    ``_handle_zeros_in_scale`` convention. Returns float32.

    Parameters
    ----------
    raw : numpy.ndarray
        ``(n, N_DESCRIPTORS)`` raw descriptors.
    scaler_params : dict
        Parameters from :func:`fit_scaler`.

    Returns
    -------
    numpy.ndarray
        Imputed, standard-scaled float32 matrix.
    """
    median = np.asarray(scaler_params["median"], dtype=np.float64)
    mean = np.asarray(scaler_params["mean"], dtype=np.float64)
    scale = np.asarray(scaler_params["scale"], dtype=np.float64)
    safe_scale = np.where(scale > 0, scale, 1.0)
    imputed = np.where(np.isfinite(raw), raw.astype(np.float64), median[None, :])
    return ((imputed - mean[None, :]) / safe_scale[None, :]).astype(np.float32)
