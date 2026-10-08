"""Fit the physchem scaler shipped with the package (maintainers).

    python scripts/fit_physchem_scaler.py [N_MOLECULES] [N_JOBS]

Computes the RDKit descriptors of the canonical reference library's molecules
(a seeded sample of ``N_MOLECULES``, default all), fits the per-column median
imputer and StandardScaler on them, and prints the JSON to stdout. Redirect it
to ``src/eosquality/library/physchem_scaler.json`` to replace the shipped copy
(then recompute ``PAIR_MEDIAN`` with ``scripts/physchem_pair_median.py``).
"""

import json
import sys

import numpy as np
import sklearn
from rdkit import __version__ as rdkit_version
from sklearn.preprocessing import StandardScaler

from eosquality.library.physchem import (
    DESCRIPTOR_NAMES,
    N_DESCRIPTORS,
    compute_physchem_raw,
)
from eosquality.library.reference import ReferenceLibrary


def fit_scaler(raw: np.ndarray) -> dict:
    """Fit the per-column median imputer + StandardScaler.

    Non-finite entries (``NaN``, ``±inf``) are replaced with the
    per-column median (computed over finite values). StandardScaler is
    then fit on the imputed matrix. Returns a JSON-serialisable dict
    with every parameter the physchem domain needs.

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
    # so the scaling is well-defined.
    all_nan = np.isnan(median)
    if all_nan.any():
        bad = [DESCRIPTOR_NAMES[i] for i in np.flatnonzero(all_nan)]
        print(
            f"physchem | {int(all_nan.sum())} descriptor(s) had no finite "
            f"values across the reference; imputing with 0 → {bad}",
            file=sys.stderr,
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
        "rdkit_version": rdkit_version,
        "sklearn_version": sklearn.__version__,
    }


def main(n_molecules: int | None = None, n_jobs: int = -1) -> None:
    """Fit the scaler on the library and print it as JSON.

    Parameters
    ----------
    n_molecules : int, optional
        Seeded sample size (default: every molecule).
    n_jobs : int, optional
        Worker processes for the descriptors (``-1``: all cores).
    """
    smiles = ReferenceLibrary.load().smiles
    if n_molecules and n_molecules < len(smiles):
        picked = np.random.default_rng(0).choice(len(smiles), n_molecules, False)
        smiles = [smiles[i] for i in sorted(picked)]
    raw = compute_physchem_raw(smiles, n_jobs=n_jobs)
    print(json.dumps(fit_scaler(raw), indent=2))


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:3]]
    main(*args)
