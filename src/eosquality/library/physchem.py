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

The descriptor loop is parallelized with ``multiprocessing.Pool``; the
progress bar is a ``rich.Progress`` mirroring the style used in
``library/download.py``. The step is resume-cached: if both artifacts
already exist, the call is a no-op unless ``force=True``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pathlib
import time
from typing import Iterable

import numpy as np
import sklearn
from rdkit import Chem
from rdkit import __version__ as _RDKIT_VERSION
from rdkit.Chem import Descriptors
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.preprocessing import StandardScaler

from eosquality.utils.logging import logger


PHYSCHEM_SCALED_FILE = "physchem_scaled.npy"
PHYSCHEM_SCALER_FILE = "physchem_scaler.json"


# RDKit's canonical descriptor list — (name, callable) tuples. Captured
# at import time so every worker (under multiprocessing 'spawn') sees
# the same ordering after re-importing this module.
DESCRIPTOR_FNS: list[tuple[str, callable]] = list(Descriptors._descList)
DESCRIPTOR_NAMES: list[str] = [name for name, _ in DESCRIPTOR_FNS]
N_DESCRIPTORS: int = len(DESCRIPTOR_FNS)


# Always show the progress bar on stderr regardless of the global logger
# verbosity. Long step, the user explicitly asked for visible progress.
_console = Console(stderr=True, highlight=False)


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


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]physchem[/bold cyan]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    )


def compute_physchem_raw(
    smiles: Iterable[str],
    *,
    n_jobs: int | None = None,
    chunksize: int = 256,
) -> np.ndarray:
    """Compute the ``(n, N_DESCRIPTORS)`` raw float32 descriptor matrix.

    Parallelised with ``multiprocessing.Pool.imap`` so rows are
    assembled in input order. ``n_jobs=None`` uses every CPU. Raw
    values include NaN / ±inf — imputation happens in :func:`fit_scaler`
    and :func:`apply_scaler`, not here.
    """
    smiles_list = list(smiles)
    n = len(smiles_list)
    if n == 0:
        return np.zeros((0, N_DESCRIPTORS), dtype=np.float32)

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, n))

    out = np.empty((n, N_DESCRIPTORS), dtype=np.float32)

    with _build_progress() as progress:
        task_id = progress.add_task("descriptors", total=n)
        with mp.Pool(processes=n_jobs) as pool:
            for i, row in enumerate(
                pool.imap(_compute_one, smiles_list, chunksize=chunksize)
            ):
                out[i] = row
                progress.update(task_id, advance=1)
    return out


# ---------------------------------------------------------------------------
# Scaler (imputer + StandardScaler) — fit & apply
# ---------------------------------------------------------------------------


def fit_scaler(raw: np.ndarray) -> dict:
    """Fit the per-column median imputer + StandardScaler.

    Non-finite entries (``NaN``, ``±inf``) are replaced with the
    per-column median (computed over finite values). StandardScaler is
    then fit on the imputed matrix. Returns a JSON-serialisable dict
    with every parameter needed by :func:`apply_scaler`.
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


def apply_scaler(raw: np.ndarray, scaler_params: dict) -> np.ndarray:
    """Apply impute → standard-scale using persisted parameters.

    Mirrors :func:`fit_scaler`: non-finite entries replaced by
    ``scaler_params["median"]``, then ``(x - mean) / scale``. Columns
    whose ``scale`` is 0 (constant in the reference) are divided by 1
    to avoid division-by-zero, matching scikit-learn's internal
    ``_handle_zeros_in_scale`` convention. Returns float32.
    """
    median = np.asarray(scaler_params["median"], dtype=np.float64)
    mean = np.asarray(scaler_params["mean"], dtype=np.float64)
    scale = np.asarray(scaler_params["scale"], dtype=np.float64)
    safe_scale = np.where(scale > 0, scale, 1.0)
    imputed = np.where(np.isfinite(raw), raw.astype(np.float64), median[None, :])
    return ((imputed - mean[None, :]) / safe_scale[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _read_smiles(index_dir: pathlib.Path, max_samples: int | None) -> list[str]:
    smiles_path = index_dir / "smiles.csv"
    if not smiles_path.is_file():
        raise FileNotFoundError(
            f"Cannot compute physchem: {smiles_path} not found. "
            "Build the Morgan index first (eosquality build)."
        )
    smiles: list[str] = []
    with open(smiles_path) as f:
        header = f.readline().strip()
        if header.lower() != "smiles":
            raise ValueError(
                f"Unexpected header in {smiles_path}: {header!r} (want 'smiles')."
            )
        for line in f:
            s = line.strip()
            if s:
                smiles.append(s)
    if max_samples is not None and max_samples > 0:
        smiles = smiles[:max_samples]
    return smiles


def fit_and_save_physchem(
    index_dir: str | pathlib.Path,
    *,
    max_samples: int | None = None,
    n_jobs: int | None = None,
    force: bool = False,
) -> None:
    """Compute, scale, and persist physchem artifacts in ``index_dir``.

    Reads ``index_dir/smiles.csv``, computes the raw descriptor matrix
    in parallel with a progress bar, fits the imputer + StandardScaler,
    and writes ``physchem_scaled.npy`` (float16) and
    ``physchem_scaler.json`` into ``index_dir``. No-op if both are
    already present and ``force`` is False.
    """
    index_dir = pathlib.Path(index_dir)
    scaled_path = index_dir / PHYSCHEM_SCALED_FILE
    scaler_path = index_dir / PHYSCHEM_SCALER_FILE

    if not force and scaled_path.is_file() and scaler_path.is_file():
        logger.info(
            f"physchem | artifacts already present in {index_dir} — skipped "
            "(pass force=True to recompute)"
        )
        return

    t0 = time.perf_counter()
    smiles = _read_smiles(index_dir, max_samples)
    n = len(smiles)
    if n == 0:
        raise ValueError(f"No SMILES read from {index_dir / 'smiles.csv'}.")

    if n_jobs is None:
        n_jobs_resolved = os.cpu_count() or 1
    else:
        n_jobs_resolved = max(1, n_jobs)
    logger.info(
        f"physchem | computing {N_DESCRIPTORS} descriptors × {n:,} molecules "
        f"| n_jobs={n_jobs_resolved}"
    )

    raw = compute_physchem_raw(smiles, n_jobs=n_jobs_resolved)
    non_finite_rows = int((~np.isfinite(raw)).any(axis=1).sum())
    logger.info(
        f"physchem | raw matrix ready | shape={raw.shape} "
        f"| rows with any non-finite descriptor: {non_finite_rows:,} "
        f"({100 * non_finite_rows / n:.2f}%)"
    )

    scaler_params = fit_scaler(raw)
    scaled = apply_scaler(raw, scaler_params).astype(np.float16)

    np.save(scaled_path, scaled)
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)

    duration = time.perf_counter() - t0
    logger.success(
        f"physchem | saved → {index_dir} | "
        f"scaled={scaled_path.name} (float16) scaler={scaler_path.name} "
        f"| {duration:.2f}s"
    )


def load_physchem(index_dir: str | pathlib.Path) -> dict:
    """Read the two physchem artifacts from ``index_dir``.

    Returns ``{"scaled": ndarray, "scaler_params": dict}``.
    Raises ``FileNotFoundError`` if either file is missing —
    callers that want a "load if present" check should test
    ``scaled_path.is_file()`` etc. themselves.
    """
    index_dir = pathlib.Path(index_dir)
    scaled_path = index_dir / PHYSCHEM_SCALED_FILE
    scaler_path = index_dir / PHYSCHEM_SCALER_FILE
    for p in (scaled_path, scaler_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing physchem artifact: {p}")
    scaled = np.load(scaled_path)
    with open(scaler_path) as f:
        scaler_params = json.load(f)
    return {"scaled": scaled, "scaler_params": scaler_params}
