"""TrainingFitState: per-column training sets + one Morgan index per column.

Layout under ``<root>/training_sets/`` (``<root>`` is ``training_mode/`` in
an :class:`~eosquality.quality.ErsiliaQuality` artifacts folder)::

    metadata.json        # training_format_version, column order
    columns.json         # per column: folder, n, y_kind, has_y, has_pred
    arrays.npz           # per column: ids, y, pred (when present)
    indices/c000/ …      # one VectorIndex folder per column (vector_index.h5,
                         # knn_*.npy, smiles.csv, metadata.json)

Each per-column index is built with :meth:`VectorIndex.build`, whose
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
# ARTIFACT_FORMAT_VERSION so reference-only artifacts are unaffected.
# 2: training_distance (uncalibrated) replaced training_domain.
# 3: training_distance is the mean distance to the 5 nearest training
#    molecules, calibrated on the leave-one-out values.
# 4: training_difficulty uses UNIQUE feature set (i) only (state.json carries
#    `spearman` and `cv`, no `variant`); training-molecule queries reuse
#    their out-of-fold inputs (arrays.npz carries `oof_error`).
# 5: error models are fitted on at most MAX_FIT_MOLECULES labelled molecules
#    per column, so the per-molecule arrays (residuals, oof_*) are NaN
#    outside that subset and state.json carries `n_fit`.
# 6: the error model's inputs are four scalars (nn1_tanimoto, nn5_tanimoto,
#    ensemble_variance, surrogate_score); the MACCS data features and the
#    three KDE log-densities are gone, so density.joblib is no longer written
#    and scores/_density.py was deleted.
# 7: adds training_physchem (mean distance to the 5 nearest training
#    molecules over standardised physchem descriptors), saved under
#    training_physchem/c000/.
# 8: adds training_match (connectivity layers of the training molecules and
#    their Murcko scaffolds) under training_match/; training_distance is
#    published as trn_tanimoto.
# 9: training_physchem scales with the reference library's shipped scaler,
#    clipped to +/-10, instead of the training set's own statistics.
TRAINING_FORMAT_VERSION = 9
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
    """Build one Morgan :class:`VectorIndex` per training column (in a temp dir).

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
    for i, name in enumerate(console.track(list(columns), "Morgan index, columns")):
        column = columns[name]
        t0 = time.perf_counter()
        indices[name] = VectorIndex.build(
            column.smiles,
            pathlib.Path(workdir.name) / _folder(i),
            max_k=min(TRAINING_MAX_K, column.n - 2),
            library_name=f"training:{name}",
        )
        logger.info(
            f"training | column {name!r}: index built | n={column.n:,} | "
            f"{time.perf_counter() - t0:.1f}s"
        )
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
    """Write ``<root>/training_sets/`` (metadata, labels, per-column indices).

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
    for i, (name, column) in enumerate(state.columns.items()):
        sub = _folder(i)
        target = folder / "indices" / sub
        source = state.indices[name].index_dir
        if source.resolve() != target.resolve():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        meta_cols[name] = {
            "folder": sub,
            "n": column.n,
            "y_kind": column.y_kind,
            "has_y": column.has_y,
            "has_pred": column.has_pred,
        }
        arrays[f"{sub}__ids"] = np.asarray(column.ids, dtype=str)
        if column.y is not None:
            arrays[f"{sub}__y"] = column.y
        if column.pred is not None:
            arrays[f"{sub}__pred"] = column.pred
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
    with np.load(folder / "arrays.npz") as arrays:
        for name in meta["columns"]:
            info = meta_cols[name]
            sub = info["folder"]
            vi = VectorIndex.load(folder / "indices" / sub)
            columns[name] = TrainingColumn(
                name=name,
                smiles=vi.smiles,
                ids=arrays[f"{sub}__ids"].tolist(),
                y=arrays[f"{sub}__y"] if info["has_y"] else None,
                y_kind=info["y_kind"],
                pred=arrays[f"{sub}__pred"] if info["has_pred"] else None,
            )
            indices[name] = vi
    return TrainingFitState(
        columns=columns,
        indices=indices,
        eos_id=meta.get("eos_id", ""),
        version=meta.get("version", ""),
    )


def _folder(i: int) -> str:
    return f"c{i:03d}"
