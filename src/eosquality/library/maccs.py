"""MACCS-keys binary fingerprint matrix for the reference library.

At library-build time, compute the 166-bit MACCS keys for every
reference molecule and persist a single artifact alongside the Morgan
and physchem files:

- ``maccs.npy`` — ``(n_ref, 166)`` uint8; one row per reference
  molecule, one column per MACCS key (positions 1–166 of RDKit's
  ``MACCSkeys.GenMACCSKeys`` output — the leading bit 0 of the 167-bit
  vector is always zero by convention and is dropped here).

No scaler is persisted because the keys are binary: downstream code can
read the matrix directly with ``np.load`` and use it as-is.

The descriptor loop is parallelised with ``multiprocessing.Pool``; the
progress bar mirrors the style used in ``library/physchem.py`` and
``library/download.py``. The step is resume-cached: if ``maccs.npy``
already exists, the call is a no-op unless ``force=True``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pathlib
import time
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
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

from eosquality.utils.logging import logger


MACCS_FILE = "maccs.npy"

# RDKit returns a 167-bit ExplicitBitVect; bit 0 is always 0 by the MACCS
# convention and is dropped here so the persisted matrix has the 166
# meaningful keys.
MACCS_WIDTH_RAW: int = 167
N_MACCS: int = 166

# Always show the progress bar on stderr regardless of the global logger
# verbosity — the user explicitly asked for visible progress.
_console = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _compute_one(smi: str) -> np.ndarray:
    """Compute the 166-bit MACCS row for one SMILES.

    Returns a ``(166,)`` uint8 row with values in ``{0, 1}``. On any
    failure (parse error, MACCS exception) the row is all zeros so
    downstream consumers can rely on uniform shape and dtype.
    Module-level so it is picklable for ``multiprocessing.Pool``.
    """
    row_full = np.zeros(MACCS_WIDTH_RAW, dtype=np.uint8)
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return row_full[1:]
    if mol is None:
        return row_full[1:]
    try:
        bv = MACCSkeys.GenMACCSKeys(mol)
        ConvertToNumpyArray(bv, row_full)
    except Exception:
        return np.zeros(N_MACCS, dtype=np.uint8)
    return row_full[1:]


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]maccs[/bold cyan]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    )


def compute_maccs(
    smiles: Iterable[str],
    *,
    n_jobs: int | None = None,
    chunksize: int = 1024,
) -> np.ndarray:
    """Compute the ``(n, 166)`` uint8 MACCS matrix.

    Parallelised with ``multiprocessing.Pool.imap`` so rows are
    assembled in input order. ``n_jobs=None`` uses every CPU. MACCS
    computation is much cheaper per molecule than the physchem
    descriptors, so the default chunksize is larger to keep IPC
    overhead from dominating.
    """
    smiles_list = list(smiles)
    n = len(smiles_list)
    if n == 0:
        return np.zeros((0, N_MACCS), dtype=np.uint8)

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, n))

    out = np.empty((n, N_MACCS), dtype=np.uint8)

    with _build_progress() as progress:
        task_id = progress.add_task("maccs", total=n)
        with mp.Pool(processes=n_jobs) as pool:
            for i, row in enumerate(
                pool.imap(_compute_one, smiles_list, chunksize=chunksize)
            ):
                out[i] = row
                progress.update(task_id, advance=1)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _read_smiles(index_dir: pathlib.Path, max_samples: int | None) -> list[str]:
    smiles_path = index_dir / "smiles.csv"
    if not smiles_path.is_file():
        raise FileNotFoundError(
            f"Cannot compute MACCS: {smiles_path} not found. "
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


def fit_and_save_maccs(
    index_dir: str | pathlib.Path,
    *,
    max_samples: int | None = None,
    n_jobs: int | None = None,
    force: bool = False,
) -> None:
    """Compute and persist the MACCS matrix in ``index_dir``.

    Reads ``index_dir/smiles.csv``, computes the binary key matrix in
    parallel with a progress bar, and writes ``maccs.npy`` into
    ``index_dir``. No-op if the file is already present and ``force``
    is False.
    """
    index_dir = pathlib.Path(index_dir)
    maccs_path = index_dir / MACCS_FILE

    if not force and maccs_path.is_file():
        logger.info(
            f"maccs | artifact already present in {index_dir} — skipped "
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
        f"maccs | computing {N_MACCS} keys × {n:,} molecules | n_jobs={n_jobs_resolved}"
    )

    matrix = compute_maccs(smiles, n_jobs=n_jobs_resolved)
    empty_rows = int((matrix.sum(axis=1) == 0).sum())
    mean_on_bits = float(matrix.sum(axis=1).mean())
    logger.info(
        f"maccs | matrix ready | shape={matrix.shape} "
        f"| empty rows (parse / compute failure): {empty_rows:,} "
        f"({100 * empty_rows / n:.2f}%) "
        f"| mean on-bits per molecule: {mean_on_bits:.1f}"
    )

    np.save(maccs_path, matrix)

    duration = time.perf_counter() - t0
    logger.success(
        f"maccs | saved → {index_dir} | maccs={maccs_path.name} | {duration:.2f}s"
    )


def load_maccs(index_dir: str | pathlib.Path) -> np.ndarray:
    """Read ``maccs.npy`` from ``index_dir``.

    Returns the ``(n, 166)`` uint8 matrix. Raises ``FileNotFoundError``
    if the file is missing — callers that want a "load if present"
    check should test ``maccs_path.is_file()`` themselves.
    """
    index_dir = pathlib.Path(index_dir)
    maccs_path = index_dir / MACCS_FILE
    if not maccs_path.is_file():
        raise FileNotFoundError(f"Missing MACCS artifact: {maccs_path}")
    return np.load(maccs_path)
