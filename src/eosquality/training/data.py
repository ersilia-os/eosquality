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
from functools import cached_property

import numpy as np
import pandas as pd

from eosquality.exceptions import SchemaError
from eosquality.scores._helpers import _standardize
from eosquality.utils.logging import logger

# Columns with fewer training molecules than this are skipped: their
# leave-one-out calibration would have steps coarser than ~5%.
MIN_TRAINING_MOLECULES = 20

_SMILES_COLUMNS = ("smiles", "input")


@dataclass
class TrainingColumn:
    """The standardised training set of one model output column."""

    name: str
    smiles: list[str]  # standardised, unique, sorted
    ids: list[str]  # the file's `key` per molecule, or "<column>:<row>"
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

    columns: dict[str, TrainingColumn] = {}
    standardized: dict[str, str | None] = {}  # shared: panels repeat molecules
    for name in order:
        if name not in files:
            continue
        column = _load_column(name, pd.read_csv(files[name]), standardized)
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


def _load_column(
    name: str, df: pd.DataFrame, standardized: dict[str, str | None] | None = None
) -> TrainingColumn:
    smiles_col = next((c for c in _SMILES_COLUMNS if c in df.columns), None)
    if smiles_col is None:
        raise SchemaError(
            f"Training file for column {name!r} needs a 'smiles' column "
            f"(found {list(df.columns)})."
        )
    cache = {} if standardized is None else standardized
    std = df[smiles_col].map(
        lambda s: cache[s] if s in cache else cache.setdefault(s, _standardize(s))
    )
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
    smiles = list(grouped.groups.keys())
    first_ids = grouped["id"].first().reindex(smiles).tolist()
    n_dupes = len(table) - len(smiles)
    if n_dupes:
        logger.info(f"training | column {name!r}: {n_dupes} duplicate rows merged")
    return TrainingColumn(
        name=name,
        smiles=smiles,
        ids=first_ids,
        n_unparsable=n_bad,
    )
