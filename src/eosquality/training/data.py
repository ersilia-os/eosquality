"""Read, validate and standardise per-column training sets.

Input contract: a folder with one CSV per model output column,
``<folder>/<column>.csv``, holding a ``smiles`` column (``input`` is accepted
as an alias), optionally a numeric ``y`` column (``value`` is accepted as an
alias) and a ``key`` column. When
the model's output columns are known (from the reference predictions), every
file must name one of them.

Optionally, a ``training_predictions`` CSV in the usual Ersilia output shape
(``input`` + one column per model output) carries the model's own
predictions on the training molecules; rows are matched by standardised
SMILES.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eosquality.exceptions import SchemaError
from eosquality.scores._helpers import _standardize
from eosquality.shared.metadata import _detect_kind
from eosquality.utils.logging import logger

# Columns with fewer training molecules than this are skipped: their
# leave-one-out calibration would have steps coarser than ~5%.
MIN_TRAINING_MOLECULES = 20

_SMILES_COLUMNS = ("smiles", "input")
_LABEL_COLUMNS = ("y", "value")


@dataclass
class TrainingColumn:
    """The standardised training set of one model output column."""

    name: str
    smiles: list[str]  # standardised, unique, in file order of first occurrence
    ids: list[str]  # the file's `key` per molecule, or "<column>:<row>"
    y: np.ndarray | None  # (n,) float labels, NaN where missing; None if no y column
    y_kind: str | None  # "binary" | "count" | "continuous"; None without y
    pred: np.ndarray | None = None  # (n,) model predictions on these molecules
    n_unparsable: int = 0  # file rows dropped: SMILES missing or unparsable
    n_conflicting: int = 0  # molecules listed more than once with different labels

    @property
    def n(self) -> int:
        """Number of (unique, standardised) training molecules.

        Returns
        -------
        int
        """
        return len(self.smiles)

    @property
    def has_y(self) -> bool:
        """Whether the training file had a ``y`` column.

        Returns
        -------
        bool
        """
        return self.y is not None

    @property
    def has_pred(self) -> bool:
        """Whether model predictions were matched.

        Returns
        -------
        bool
        """
        return self.pred is not None


def load_training(
    folder: str | pathlib.Path,
    output_columns: list[str] | None = None,
    predictions: str | pathlib.Path | pd.DataFrame | None = None,
) -> dict[str, TrainingColumn]:
    """Load every ``<column>.csv`` in ``folder`` into a :class:`TrainingColumn`.

    ``output_columns`` are the model's output columns, when known; a file
    naming anything else raises :class:`SchemaError`. Without them (a
    training-only fit) every file is taken as a column. Unparsable SMILES
    are dropped; duplicate molecules (after standardisation) are merged —
    binary labels by majority (ties → 1), other labels by median. Columns
    left with fewer than ``MIN_TRAINING_MOLECULES`` molecules are skipped.
    Returns columns in ``output_columns`` order (file-name order otherwise).

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder with one ``<column>.csv`` per output column.
    output_columns : list of str, optional
        The model's output columns, when known.
    predictions : str, pathlib.Path or pandas.DataFrame, optional
        The model's predictions on the training molecules.

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
    pred_lookup = (
        _prediction_lookup(predictions, order) if predictions is not None else None
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
        if pred_lookup is not None:
            column.pred = _match_predictions(column, pred_lookup)
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
    y_col = next((c for c in _LABEL_COLUMNS if c in df.columns), None)
    has_y = y_col is not None
    if has_y and not pd.api.types.is_numeric_dtype(df[y_col]):
        raise SchemaError(
            f"Training file for column {name!r}: {y_col!r} must be numeric."
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
    if has_y:
        table["y"] = df[y_col].astype(float)
    table = table[table["smiles"].notna()]

    y_kind = _detect_kind(table["y"]) if has_y else None
    grouped = table.groupby("smiles", sort=False)
    smiles = list(grouped.groups.keys())
    first_ids = grouped["id"].first().reindex(smiles).tolist()
    y = None
    if has_y:
        if y_kind == "binary":
            merged = grouped["y"].mean().reindex(smiles)
            y = np.where(merged.isna(), np.nan, (merged >= 0.5).astype(float)).astype(
                float
            )
        else:
            y = grouped["y"].median().reindex(smiles).to_numpy(dtype=float)
        conflicts = int((grouped["y"].nunique() > 1).sum())
        if conflicts:
            logger.info(
                f"training | column {name!r}: {conflicts} molecules appear more "
                "than once with different labels (merged)"
            )
    n_dupes = len(table) - len(smiles)
    if n_dupes:
        logger.info(f"training | column {name!r}: {n_dupes} duplicate rows merged")
    return TrainingColumn(
        name=name,
        smiles=smiles,
        ids=first_ids,
        y=y,
        y_kind=y_kind,
        n_unparsable=n_bad,
        n_conflicting=conflicts if has_y else 0,
    )


def _prediction_lookup(
    predictions: str | pathlib.Path | pd.DataFrame, output_columns: list[str]
) -> pd.DataFrame:
    df = (
        predictions
        if isinstance(predictions, pd.DataFrame)
        else pd.read_csv(predictions)
    )
    smiles_col = next((c for c in _SMILES_COLUMNS if c in df.columns), None)
    if smiles_col is None:
        raise SchemaError("Training predictions need an 'input' (SMILES) column.")
    cols = [c for c in output_columns if c in df.columns]
    if not cols:
        raise SchemaError(
            "Training predictions contain none of the model's output columns."
        )
    out = df[cols].copy()
    out.index = df[smiles_col].map(_standardize)
    out = out[out.index.notna()]
    return out.groupby(level=0).mean()


def _match_predictions(
    column: TrainingColumn, lookup: pd.DataFrame
) -> np.ndarray | None:
    if column.name not in lookup.columns:
        return None
    pred = lookup[column.name].reindex(column.smiles).to_numpy(dtype=float)
    missing = int(np.isnan(pred).sum())
    if missing:
        logger.warning(
            f"training | column {column.name!r}: no model prediction for "
            f"{missing} of {column.n} training molecules"
        )
    return pred
