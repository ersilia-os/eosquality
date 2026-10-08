"""TrainingFitState: per-column training sets + a Morgan index per distinct set.

Layout under ``<root>/training_sets/`` (``<root>`` is ``training_mode/`` in
an :class:`~eosquality.quality.ErsiliaQuality` artifacts folder)::

    metadata.json        # training_format_version, column order
    columns.json         # per column: its index folder, n
    arrays.npz           # per column: ids; per index: every training molecule (c000__all)
    indices/c000/ …      # one VectorIndex folder per distinct molecule set
                         # (vector_index.h5, knn_distances.npy, smiles.csv, metadata.json)

A column's index holds one molecule per distinct Morgan fingerprint
(``TrainingColumn.smiles``); ``all_smiles`` keeps the rest.

Columns measured on the same molecules (one screening panel) share an index.
Each index is built with :meth:`VectorIndex.build`, whose
identity-stripped self-kNN gives every training molecule its leave-one-out
nearest-training similarity — the calibration table for training scores.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from eosquality.exceptions import ArtifactVersionError
from eosquality.training.data import TrainingColumn
from eosquality.utils import console
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

SUBFOLDER = "training_sets"
# Bump when the meaning or layout of training_sets/ changes. Independent of
# ARTIFACT_FORMAT_VERSION so reference-only artifacts are unaffected. History:
# git log.
TRAINING_FORMAT_VERSION = 14
# Neighbours precomputed per training molecule (capped by column size).
TRAINING_MAX_K = 10


@dataclass
class TrainingFitState:
    """Training sets (one per output column) and their fingerprint indices."""

    columns: dict[str, TrainingColumn]
    indices: dict[str, VectorIndex]
    eos_id: str = ""
    version: str = ""
    # Keeps a fit-time build directory alive until the state is saved/dropped.
    _workdir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    @property
    def column_names(self) -> list[str]:
        """Output columns with a training set, in order.

        Returns
        -------
        list of str
        """
        return list(self.columns)


def fit_training(
    columns: dict[str, TrainingColumn], *, eos_id: str, version: str
) -> TrainingFitState:
    """Build a Morgan :class:`VectorIndex` per distinct training set (in a temp dir).

    Columns measured on the same molecules (one screening panel) share one
    index.

    Parameters
    ----------
    columns : dict of str to TrainingColumn
        Loaded training sets.
    eos_id, version : str
        Model id and version, stored with the training state.

    Returns
    -------
    TrainingFitState
    """
    workdir = tempfile.TemporaryDirectory(prefix="eosquality_training_")
    indices: dict[str, VectorIndex] = {}
    built: dict[str, VectorIndex] = {}  # molecule-set signature → index
    try:
        for name in console.track(list(columns), "Morgan index, columns"):
            column = columns[name]
            if column.signature in built:
                indices[name] = built[column.signature]
                logger.info(f"training | column {name!r}: shares an index built before")
                continue
            t0 = time.perf_counter()
            built[column.signature] = indices[name] = VectorIndex.build(
                column.smiles,
                pathlib.Path(workdir.name) / _folder(len(built)),
                max_k=min(TRAINING_MAX_K, column.n - 2),
            )
            logger.info(
                f"training | column {name!r}: index built | n={column.n:,} | "
                f"{time.perf_counter() - t0:.1f}s"
            )
    except BaseException:
        workdir.cleanup()  # do not leave a half-built index folder behind
        raise
    return TrainingFitState(
        columns=columns,
        indices=indices,
        eos_id=eos_id,
        version=version,
        _workdir=workdir,
    )


def save_training_state(
    state: TrainingFitState, root: str | pathlib.Path
) -> pathlib.Path:
    """Write ``<root>/training_sets/`` (metadata, ids, per-column indices).

    Parameters
    ----------
    state : TrainingFitState
        Fitted training state.
    root : str or pathlib.Path
        Folder to write into.

    Returns
    -------
    pathlib.Path
        The ``training_sets`` folder.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)
    meta_cols, arrays = {}, {}
    saved: dict[str, str] = {}  # molecule-set signature → index folder
    for i, (name, column) in enumerate(state.columns.items()):
        if column.signature not in saved:
            sub = saved[column.signature] = _folder(len(saved))
            target = folder / "indices" / sub
            source = state.indices[name].index_dir
            if source.resolve() != target.resolve():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
        meta_cols[name] = {"folder": saved[column.signature], "n": column.n}
        arrays[f"{_folder(i)}__ids"] = np.asarray(column.ids, dtype=str)
        arrays[f"{saved[column.signature]}__all"] = np.asarray(
            column.all_smiles, dtype=str
        )
    np.savez(folder / "arrays.npz", **arrays)
    with open(folder / "columns.json", "w") as f:
        json.dump(meta_cols, f, indent=2)
    with open(folder / "metadata.json", "w") as f:
        json.dump(
            {
                "training_format_version": TRAINING_FORMAT_VERSION,
                "eos_id": state.eos_id,
                "version": state.version,
                "columns": state.column_names,
            },
            f,
            indent=2,
        )
    return folder


def load_training_state(root: str | pathlib.Path) -> TrainingFitState:
    """Read ``<root>/training_sets/``; rejects other training format versions.

    Parameters
    ----------
    root : str or pathlib.Path
        Folder that contains ``training_sets/``.

    Returns
    -------
    TrainingFitState
        The loaded state, with one memory-mapped index per column.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    with open(folder / "metadata.json") as f:
        meta = json.load(f)
    version = meta.get("training_format_version")
    if version != TRAINING_FORMAT_VERSION:
        raise ArtifactVersionError(
            f"Training artifacts at {folder} use training format {version}; this "
            f"eosquality install reads format {TRAINING_FORMAT_VERSION}. Refit."
        )
    with open(folder / "columns.json") as f:
        meta_cols = json.load(f)
    columns: dict[str, TrainingColumn] = {}
    indices: dict[str, VectorIndex] = {}
    loaded: dict[str, VectorIndex] = {}  # index folder → index, shared by columns
    with np.load(folder / "arrays.npz") as arrays:
        for i, name in enumerate(meta["columns"]):
            sub = meta_cols[name]["folder"]
            if sub not in loaded:
                loaded[sub] = VectorIndex.load(folder / "indices" / sub)
            columns[name] = TrainingColumn(
                name=name,
                smiles=loaded[sub].smiles,
                ids=arrays[f"{_folder(i)}__ids"].tolist(),
                all_smiles=arrays[f"{sub}__all"].tolist(),
            )
            indices[name] = loaded[sub]
    return TrainingFitState(
        columns=columns,
        indices=indices,
        eos_id=meta.get("eos_id", ""),
        version=meta.get("version", ""),
    )


def _folder(i: int) -> str:
    return f"c{i:03d}"
