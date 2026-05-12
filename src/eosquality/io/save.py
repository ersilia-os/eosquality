"""Serialize a FitState to a folder of artifact files."""

import dataclasses
import json
import pathlib

import joblib
import numpy as np

from eosquality.reference.fit_state import FitState
from eosquality.utils.logging import logger


def save(fit_state: FitState, path: str | pathlib.Path) -> pathlib.Path:
    """Persist fit_state as a folder of artifact files.

    The folder contains:
    - ``config.json``               — ErsiliaQualityConfig + vector_index_path
    - ``schema.json``               — Schema column specs (human-readable)
    - ``reference_report.json``     — ReferenceReport diagnostics (human-readable)
    - ``metadata.json``             — FitMetadata provenance
    - ``scaler.json``               — eosframes scaler params (human-readable)
    - ``reference_ids.joblib``      — Reference row index labels
    - ``reference_repr.npy``        — Scaled reference feature array
    - ``reference_knn_distances.npy`` — Self-kNN output-space distances (n_ref, k)
    - ``reference_knn_indices.npy`` — Self-kNN FP-selected indices (n_ref, k)

    Parameters
    ----------
    fit_state:
        The FitState produced by ErsiliaQuality.fit().
    path:
        Directory to write. Created if it does not exist.

    Returns
    -------
    pathlib.Path
        The resolved folder path.
    """
    folder = pathlib.Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving artifacts → {folder}")

    # Human-readable JSON artifacts
    config_dict = dataclasses.asdict(fit_state.config)
    config_dict["vector_index_path"] = fit_state.vector_index_path
    with open(folder / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.debug("  config.json")

    with open(folder / "schema.json", "w") as f:
        json.dump(dataclasses.asdict(fit_state.schema), f, indent=2)
    logger.debug("  schema.json")

    with open(folder / "reference_report.json", "w") as f:
        json.dump(dataclasses.asdict(fit_state.reference_report), f, indent=2)
    logger.debug("  reference_report.json")

    with open(folder / "metadata.json", "w") as f:
        json.dump(dataclasses.asdict(fit_state.metadata), f, indent=2)
    logger.debug("  metadata.json")

    # eosframes scaler params: a plain dict that round-trips through JSON.
    with open(folder / "scaler.json", "w") as f:
        json.dump(fit_state.preprocess_state["scaler_params"], f, indent=2)
    logger.debug("  scaler.json")

    # Arbitrary Python objects (index labels)
    joblib.dump(fit_state.reference_ids, folder / "reference_ids.joblib")
    logger.debug("  reference_ids.joblib")

    # Numpy arrays
    np.save(folder / "reference_repr.npy", fit_state.reference_repr)
    logger.debug(f"  reference_repr.npy | shape={fit_state.reference_repr.shape}")
    np.save(folder / "reference_knn_distances.npy", fit_state.reference_knn_distances)
    logger.debug(f"  reference_knn_distances.npy | shape={fit_state.reference_knn_distances.shape}")
    np.save(folder / "reference_knn_indices.npy", fit_state.reference_knn_indices)
    logger.debug(f"  reference_knn_indices.npy | shape={fit_state.reference_knn_indices.shape}")

    logger.debug(f"All artifacts written → {folder}")
    return folder
