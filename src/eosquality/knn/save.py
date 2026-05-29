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
    each score persists its own reduction. The scaled reference matrix
    lives once under ``<root>/shared/reference_repr.npy``; consumers
    that need it (Consistency) read it from there. The vector-index
    path is not persisted either; it is resolved at run time by
    library_id.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / STATE_FILE, "w") as f:
        json.dump({"k": int(state.k)}, f, indent=2)

    logger.debug(f"  knn/ | k={state.k}")
    return folder
