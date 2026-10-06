"""Scaffold-grouped CV folds and Morgan bit matrices for training-set learners."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def _scaffold(smiles: str) -> str:
    """Murcko scaffold SMILES; ``""`` for acyclic molecules or on failure.

    RDKit fails to canonicalise some scaffolds that keep a stereo double bond
    next to a ring once the side chains are cut; those are retried without
    stereo (the ring scaffold is the same).
    """
    try:
        with rdBase.BlockLogs():  # the failure prints an RDKit banner otherwise
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except RuntimeError:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        Chem.RemoveStereochemistry(mol)
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except RuntimeError:
            return ""


def scaffold_folds(smiles: list[str], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign whole Murcko-scaffold groups to ``n_folds`` balanced folds.

    Groups are shuffled, then each goes to the currently smallest fold.
    Acyclic molecules (empty scaffold), and the rare ones RDKit cannot
    scaffold, are their own singleton groups, so they
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
    scaffolds = [_scaffold(s) or f"__single_{i}" for i, s in enumerate(smiles)]
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


def cv_folds(
    smiles: list[str],
    labelled: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    max_fold_fraction: float = 0.4,
) -> tuple[np.ndarray, str]:
    """Scaffold folds when they make a usable split, else seeded random folds.

    Scaffold-grouped folds give honest out-of-fold errors, but a congeneric
    training set (one or a few Murcko scaffolds) cannot fill ``n_folds``
    folds, and one dominant scaffold leaves a fold so large that its
    surrogate is trained on a handful of molecules. In those cases the
    molecules are assigned to folds at random instead.

    Parameters
    ----------
    smiles : list of str
        Valid SMILES.
    labelled : numpy.ndarray
        ``(n,)`` bool, the molecules the folds must split well.
    n_folds : int, optional
        Number of folds.
    seed : int, optional
        Seed for the group order and the random fallback.
    max_fold_fraction : float, optional
        Largest share of the labelled molecules one scaffold fold may hold.

    Returns
    -------
    tuple of (numpy.ndarray, str)
        Fold id per molecule, and ``"scaffold"`` or ``"random"``.
    """
    folds = scaffold_folds(smiles, n_folds, seed)
    counts = np.bincount(folds[labelled], minlength=n_folds)
    n_labelled = int(labelled.sum())
    if (counts > 0).sum() == n_folds and counts.max() <= max_fold_fraction * n_labelled:
        return folds, "scaffold"
    rng = np.random.default_rng(seed)
    return rng.permutation(len(smiles)) % n_folds, "random"


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
    with rdBase.BlockLogs():
        for i, s in enumerate(smiles):
            out[i] = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return out
