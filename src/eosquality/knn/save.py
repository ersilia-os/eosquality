"""Serialize a KnnFitState into <root>/knn/."""

from __future__ import annotations

import json
import pathlib

import numpy as np

from eosquality.knn.state import KnnFitState
from eosquality.utils.logging import logger


SUBFOLDER = "knn"
STATE_FILE = "state.json"
REPR_FILE = "reference_repr.npy"


def save_knn(state: KnnFitState, root: str | pathlib.Path) -> pathlib.Path:
    """Write the persisted fields of KnnFitState into ``<root>/knn/``.

    Only ``ref_repr`` and ``k`` are persisted; the fit-only fields
    (``mean_fp_distances``, ``reference_knn_indices``) are dropped —
    each score persists its own reduction. The vector-index path is
    not persisted either; it is resolved at run time by library_id.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)

    np.save(folder / REPR_FILE, state.ref_repr)
    with open(folder / STATE_FILE, "w") as f:
        json.dump({"k": int(state.k)}, f, indent=2)

    logger.debug(f"  knn/ | ref_repr shape={state.ref_repr.shape} | k={state.k}")
    return folder
