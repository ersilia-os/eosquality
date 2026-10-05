"""Scaffold-grouped CV folds and Morgan bit matrices for training-set learners."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def scaffold_folds(smiles: list[str], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign whole Murcko-scaffold groups to ``n_folds`` balanced folds.

    Groups are shuffled, then each goes to the currently smallest fold.
    Acyclic molecules (empty scaffold) are their own singleton groups, so they
    do not form one giant group.

    Parameters
    ----------
    smiles : list of str
        Valid SMILES.
    n_folds : int, optional
        Number of folds.
    seed : int, optional
        Seed for the group order.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` int fold id per molecule.
    """
    scaffolds = [
        MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or f"__acyclic_{i}"
        for i, s in enumerate(smiles)
    ]
    groups = pd.Series(range(len(smiles))).groupby(scaffolds).apply(list).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = np.empty(len(smiles), dtype=np.int64)
    sizes = np.zeros(n_folds, dtype=np.int64)
    for group in groups:
        f = int(np.argmin(sizes))
        folds[group] = f
        sizes[f] += len(group)
    return folds


def morgan_bits(smiles: list[str]) -> np.ndarray:
    """Morgan bit fingerprints (radius 2, 2048 bits) as a matrix.

    Parameters
    ----------
    smiles : list of str
        Valid SMILES.

    Returns
    -------
    numpy.ndarray
        ``(n, 2048)`` uint8.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    out = np.zeros((len(smiles), MORGAN_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles):
        out[i] = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return out
