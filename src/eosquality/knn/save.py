"""Serialize a KnnFitState into <root>/knn/."""

from __future__ import annotations

import json
import pathlib

from eosquality.knn.state import KnnFitState
from eosquality.utils.logging import logger

SUBFOLDER = "knn"
STATE_FILE = "state.json"


def save_knn(state: KnnFitState, root: str | pathlib.Path) -> pathlib.Path:
    """Write the persisted fields of KnnFitState into ``<root>/knn/``.

    Only ``k`` is persisted; the fit-only fields
    (``mean_fp_distances``, ``reference_knn_indices``) are dropped —
    each score persists its own reduction.

    Parameters
    ----------
    state : KnnFitState
        The kNN state.
    root : str or pathlib.Path
        Folder to write ``knn/`` into.

    Returns
    -------
    pathlib.Path
        The ``knn`` folder.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / STATE_FILE, "w") as f:
        json.dump({"k": int(state.k)}, f, indent=2)

    logger.debug(f"  knn/ | k={state.k}")
    return folder
