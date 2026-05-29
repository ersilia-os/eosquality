"""Read a KnnFitState from <root>/knn/."""

from __future__ import annotations

import json
import pathlib

from eosquality.knn.state import KnnFitState
from eosquality.utils.logging import logger


SUBFOLDER = "knn"
STATE_FILE = "state.json"


def load_knn(root: str | pathlib.Path) -> KnnFitState:
    """Read the persisted fields of KnnFitState from ``<root>/knn/``.

    The fit-only fields are returned as ``None``. The scaled reference
    matrix is no longer stored here; consumers read it from
    ``shared.ref_repr`` (loaded by :func:`load_shared`). The underlying
    vector index is not loaded here either — it is resolved on demand
    at run time via
    :func:`eosquality.library.identity.reference_library_path` using
    ``shared.metadata.library_id`` as the key.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Expected knn fit state at {folder}, but the folder does not exist."
        )

    with open(folder / STATE_FILE) as f:
        cfg = json.load(f)
    logger.debug(f"  knn/ | k={cfg['k']}")
    return KnnFitState(k=int(cfg["k"]))
