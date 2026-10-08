"""Read, validate and standardise per-column training sets.

Input contract: a folder with one CSV per model output column,
``<folder>/<column>.csv``, holding a ``smiles`` column (``input`` is accepted
as an alias) and optionally a ``key`` column. Any other column is ignored.
When the model's output columns are known (from the reference predictions),
every file must name one of them.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
import pandas as pd

from eosquality.exceptions import SchemaError
from eosquality.scores._helpers import _standardize_all
from eosquality.utils.logging import logger
from eosquality.utils.parallel import map_rows

# Columns with fewer training molecules than this are skipped: their
# leave-one-out calibration would have steps coarser than ~5%.
MIN_TRAINING_MOLECULES = 20

_SMILES_COLUMNS = ("smiles", "input")


@dataclass
class TrainingColumn:
    """The standardised training set of one model output column.

    Molecules with the same Morgan fingerprint (stereoisomers, charge states)
    are one point to the distance scores: ``smiles`` keeps the first of each
    (sorted order) and the indices are built on those, because a training set
    that holds many copies of a fingerprint has leave-one-out distances of
    zero for the copies. ``all_smiles`` keeps every molecule for the exact
    matches.
    """

    name: str
    smiles: list[str]  # one molecule per distinct Morgan fingerprint, sorted
    ids: list[str]  # the file's `key` per `smiles` entry, or "<column>:<row>"
    all_smiles: list[str]  # every standardised, unique training molecule, sorted
    n_unparsable: int = 0  # file rows dropped: SMILES missing or unparsable

    @property
    def n(self) -> int:
        """Number of (unique, standardised) training molecules.

        Returns
        -------
        int
        """
        return len(self.smiles)

    @cached_property
    def signature(self) -> str:
        """Identity of the molecule set: columns with the same one share their index.

        Returns
        -------
        str
            A digest of the (sorted) standardised SMILES.
        """
        return hashlib.sha1("\n".join(self.smiles).encode()).hexdigest()

    @cached_property
    def _members(self) -> frozenset[str]:
        return frozenset(self.all_smiles)

    def contains(self, smiles: list[str]) -> np.ndarray:
        """Whether each standardised SMILES is a training molecule of this column.

        Parameters
        ----------
        smiles : list of str
            Standardised SMILES.

        Returns
        -------
        numpy.ndarray
            ``(len(smiles),)`` bool; looks at every training molecule, not
            only the representatives in ``smiles``.
        """
        return np.fromiter((s in self._members for s in smiles), bool, len(smiles))

    @cached_property
    def _position(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.smiles)}

    def rows_of(self, smiles: list[str]) -> np.ndarray:
        """Position of each standardised SMILES in the training set.

        Parameters
        ----------
        smiles : list of str
            Standardised SMILES.

        Returns
        -------
        numpy.ndarray
            ``(len(smiles),)`` int, ``-1`` for a molecule that is not a
            training molecule of this column.
        """
        get = self._position.get
        return np.array([get(s, -1) for s in smiles], dtype=np.int64)


def load_training(
    folder: str | pathlib.Path,
    output_columns: list[str] | None = None,
) -> dict[str, TrainingColumn]:
    """Load every ``<column>.csv`` in ``folder`` into a :class:`TrainingColumn`.

    ``output_columns`` are the model's output columns, when known; a file
    naming anything else raises :class:`SchemaError`. Without them (a
    training-only fit) every file is taken as a column. Unparsable SMILES
    are dropped and duplicate molecules (after standardisation) are merged.
    Columns left with fewer than ``MIN_TRAINING_MOLECULES`` molecules are
    skipped.
    Returns columns in ``output_columns`` order (file-name order otherwise).

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder with one ``<column>.csv`` per output column.
    output_columns : list of str, optional
        The model's output columns, when known.

    Returns
    -------
    dict of str to TrainingColumn
    """
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Training folder not found: {folder}")
    files = {p.stem: p for p in sorted(folder.glob("*.csv"))}
    if not files:
        raise SchemaError(f"No <column>.csv files found in training folder {folder}.")
    order = list(output_columns) if output_columns is not None else list(files)
    unknown = sorted(set(files) - set(order))
    if unknown:
        raise SchemaError(
            f"Training files {unknown} do not match any output column of the model "
            f"(columns: {order})."
        )

    frames = {name: pd.read_csv(files[name]) for name in order if name in files}
    standardized = _standardize_files(frames)  # panels repeat molecules: once each
    fingerprints = _fingerprints(set(standardized.values()) - {None})
    columns: dict[str, TrainingColumn] = {}
    for name, frame in frames.items():
        column = _load_column(name, frame, standardized, fingerprints)
        if column.n < MIN_TRAINING_MOLECULES:
            logger.warning(
                f"training | column {name!r}: only {column.n} usable molecules "
                f"(< {MIN_TRAINING_MOLECULES}); skipped"
            )
            continue
        columns[name] = column

    missing = [c for c in order if c not in files]
    if missing:
        logger.info(
            f"training | no training set for {len(missing)} of "
            f"{len(order)} output columns"
        )
    if not columns:
        raise SchemaError(f"No usable training column in {folder}.")
    return columns


def _smiles_column(name: str, df: pd.DataFrame) -> str:
    """The SMILES column of a training file (``smiles`` or ``input``)."""
    column = next((c for c in _SMILES_COLUMNS if c in df.columns), None)
    if column is None:
        raise SchemaError(
            f"Training file for column {name!r} needs a 'smiles' column "
            f"(found {list(df.columns)})."
        )
    return column


def _standardize_files(frames: dict[str, pd.DataFrame]) -> dict[str, str | None]:
    """Standardise every distinct SMILES of the training files, once."""
    distinct = pd.unique(
        pd.concat([df[_smiles_column(name, df)] for name, df in frames.items()])
    )
    return dict(zip(distinct, _standardize_all(distinct), strict=True))


@cache
def _morgan_generator():
    """The Morgan fingerprint generator with the index's radius and size."""
    from rdkit.Chem import rdFingerprintGenerator

    from eosquality.vectorindex import N_BITS_DEFAULT, RADIUS_DEFAULT

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS_DEFAULT, fpSize=N_BITS_DEFAULT
    )


def _fingerprint(smi: str) -> bytes:
    """The Morgan fingerprint of one SMILES, as bytes (the SMILES if unparsable)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    return _morgan_generator().GetFingerprint(mol).ToBinary() if mol else smi.encode()


def _fingerprints(smiles: set[str]) -> dict[str, bytes]:
    """The Morgan fingerprint (the index's radius and size) of each SMILES.

    Parallel inside ``workers``.
    """
    distinct = sorted(smiles)
    out = np.empty(len(distinct), dtype=object)
    map_rows(_fingerprint, distinct, out, label="fingerprints")
    return dict(zip(distinct, out.tolist(), strict=True))


def _first_of_each_fingerprint(
    smiles: list[str], fingerprints: dict[str, bytes]
) -> list[int]:
    """Positions of the first molecule of every distinct fingerprint, in order."""
    seen: set[bytes] = set()
    keep = []
    for i, smi in enumerate(smiles):
        if fingerprints[smi] not in seen:
            seen.add(fingerprints[smi])
            keep.append(i)
    return keep


def _load_column(
    name: str,
    df: pd.DataFrame,
    standardized: dict[str, str | None],
    fingerprints: dict[str, bytes],
) -> TrainingColumn:
    std = df[_smiles_column(name, df)].map(standardized)
    n_bad = int(std.isna().sum())
    if n_bad:
        logger.info(f"training | column {name!r}: {n_bad} unparsable SMILES dropped")
    ids = (
        df["key"].astype(str)
        if "key" in df.columns
        else pd.Series([f"{name}:{i}" for i in range(len(df))], index=df.index)
    )
    table = pd.DataFrame({"smiles": std, "id": ids})
    table = table[table["smiles"].notna()]

    grouped = table.groupby("smiles", sort=True)
    all_smiles = list(grouped.groups.keys())
    first_ids = grouped["id"].first().reindex(all_smiles).tolist()
    n_dupes = len(table) - len(all_smiles)
    if n_dupes:
        logger.info(f"training | column {name!r}: {n_dupes} duplicate rows merged")
    keep = _first_of_each_fingerprint(all_smiles, fingerprints)
    if len(keep) < len(all_smiles):
        logger.info(
            f"training | column {name!r}: {len(all_smiles):,} molecules, "
            f"{len(keep):,} distinct Morgan fingerprints"
        )
    return TrainingColumn(
        name=name,
        smiles=[all_smiles[i] for i in keep],
        ids=[first_ids[i] for i in keep],
        all_smiles=all_smiles,
        n_unparsable=n_bad,
    )
