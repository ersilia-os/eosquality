"""MACCS structural keys for the reference library and for queries.

One bit layout everywhere: the 166 meaningful keys of RDKit's
``MACCSkeys.GenMACCSKeys`` (positions 1–166 of its 167-bit vector; bit 0
is always zero by convention and is dropped). The library build writes
``maccs.npy`` (``(n_ref, 166)`` uint8) next to the Morgan index; the
Signal score's MACCS backend reads reference rows from that file and
computes query rows with the same :func:`compute_maccs`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray

from eosquality.utils.parallel import map_rows

MACCS_FILE = "maccs.npy"
MACCS_WIDTH_RAW: int = 167
N_MACCS: int = 166


def _compute_one(smi: str) -> np.ndarray:
    """166-bit MACCS row for one SMILES; all zeros if it fails to parse."""
    row_full = np.zeros(MACCS_WIDTH_RAW, dtype=np.uint8)
    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), row_full)
    except Exception:  # RDKit raises a variety of types on malformed input
        row_full[:] = 0
    return row_full[1:]


def compute_maccs(
    smiles: Iterable[str],
    *,
    n_jobs: int | None = None,
    show_progress: bool | None = None,
) -> np.ndarray:
    """Compute the ``(n, 166)`` uint8 MACCS matrix, rows in input order.

    Parameters
    ----------
    smiles : iterable of str
        Input SMILES.
    n_jobs : int, optional
        Worker processes (default: in-process; ``-1``: every CPU).
    show_progress : bool, optional
        Show a progress bar; ``None`` shows it only for parallel runs.

    Returns
    -------
    numpy.ndarray
        ``(n, 166)`` uint8 bits; unparsable SMILES give an all-zero row.
    """
    smiles_list = list(smiles)
    out = np.empty((len(smiles_list), N_MACCS), dtype=np.uint8)
    return map_rows(
        _compute_one,
        smiles_list,
        out,
        label="maccs",
        n_jobs=n_jobs,
        chunksize=1024,
        show_progress=show_progress,
    )
