"""Connectivity-layer keys shared by the training and reference match scores.

The **connectivity layer** is the first 14 characters of a molecule's
InChIKey: the hash of the heavy-atom skeleton and hydrogens, blind to
stereochemistry, isotopes, charge state and mobile-hydrogen tautomerism. A
molecule's *scaffold key* is the connectivity layer of its Murcko scaffold.
Both scores (``trn_match`` / ``trn_scaffold`` and ``ref_match`` /
``ref_scaffold``) are exact set lookups of these keys: no calibration.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from eosquality.scores._base import require_file
from eosquality.utils.parallel import map_rows

KEYS_FILE = "connectivity_keys.npz"
# Characters of an InChIKey that form its connectivity layer (first block).
CONNECTIVITY_LENGTH = 14


def connectivity_layer(smiles: str) -> str:
    """InChIKey connectivity layer of a SMILES; ``""`` when it has none.

    Parameters
    ----------
    smiles : str
        A SMILES string.

    Returns
    -------
    str
        The first 14 characters of the InChIKey, or ``""`` for an empty or
        unparsable structure.
    """
    from rdkit import Chem, rdBase

    if not smiles:
        return ""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.GetNumAtoms():
            return ""
        key = Chem.MolToInchiKey(mol)
    return key[:CONNECTIVITY_LENGTH]


def _scaffold(smiles: str) -> str:
    """Murcko scaffold SMILES; ``""`` for acyclic molecules or on failure.

    RDKit fails to canonicalise some scaffolds that keep a stereo double bond
    next to a ring once the side chains are cut; those are retried without
    stereo (the ring scaffold is the same).
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold

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


def _layer_pair(smiles: str) -> tuple[str, str]:
    """Connectivity layers of a molecule and of its Murcko scaffold (picklable)."""
    return connectivity_layer(smiles), connectivity_layer(_scaffold(smiles))


def _layers(
    smiles: list[str], label: str = "InChIKey layers"
) -> tuple[np.ndarray, np.ndarray]:
    """Connectivity layers of the molecules and of their Murcko scaffolds.

    Spread over processes inside ``parallel.workers`` (the CLI); in-process
    otherwise.

    Parameters
    ----------
    smiles : list of str
        Standardised SMILES.
    label : str, optional
        Progress-bar title.

    Returns
    -------
    tuple of numpy.ndarray
        ``(molecule layers, scaffold layers)``, both length ``len(smiles)``;
        ``""`` where there is none.
    """
    pairs = np.empty(len(smiles), dtype=object)
    map_rows(_layer_pair, smiles, pairs, label=label, show_progress=True)
    return (
        np.array([m for m, _ in pairs], dtype=object),
        np.array([c for _, c in pairs], dtype=object),
    )


def _flags(layers: np.ndarray, known: np.ndarray) -> pd.Series:
    """1 / 0 for layers found in ``known``; NA where the layer is empty.

    ``known`` is the sorted array of :func:`unique_keys`, so each layer is one
    binary search (``np.isin`` would sort the whole array on every call).
    """
    layers = np.asarray(layers, dtype=known.dtype)
    present = np.char.str_len(layers) > 0
    at = np.minimum(np.searchsorted(known, layers), len(known) - 1)
    found = (known[at] == layers) if len(known) else np.zeros(len(layers), bool)
    return pd.Series(found.astype(int), dtype="Int64").mask(~present)


def unique_keys(layers: np.ndarray) -> np.ndarray:
    """Sorted unique non-empty layers, as a fixed-width byte-string array.

    A connectivity layer is 14 ASCII characters, so ``S14`` takes a quarter of
    the memory and disk of the ``U14`` that ``astype(str)`` would give.

    Parameters
    ----------
    layers : numpy.ndarray
        Layers from :func:`_layers`, ``""`` where there is none.

    Returns
    -------
    numpy.ndarray
    """
    return np.unique(layers[layers != ""]).astype(f"S{CONNECTIVITY_LENGTH}")


def save_keys(path: pathlib.Path, molecules: np.ndarray, scaffolds: np.ndarray) -> None:
    """Write the two key arrays (no pickling).

    Parameters
    ----------
    path : pathlib.Path
        Target ``.npz`` file.
    molecules, scaffolds : numpy.ndarray
        Sorted unique molecule and scaffold keys.
    """
    np.savez(path, molecules=molecules, scaffolds=scaffolds)


def load_keys(path: pathlib.Path, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the two key arrays written by :func:`save_keys`.

    Parameters
    ----------
    path : pathlib.Path
        The ``.npz`` file.
    component : str
        Name used in the error message if the file is missing.

    Returns
    -------
    tuple of numpy.ndarray
        ``(molecules, scaffolds)``.
    """
    with np.load(require_file(path, component), allow_pickle=False) as arrays:
        return arrays["molecules"], arrays["scaffolds"]
