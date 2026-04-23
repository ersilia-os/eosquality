"""Deserialize a FitState from a folder of artifact files."""

import importlib.metadata
import json
import pathlib

import joblib
import numpy as np
from packaging.version import Version

from eosquality._library import LIBRARY_ID
from eosquality.config import (
    AggregationConfig,
    BootstrapConfig,
    CoreConfig,
    DistanceConfig,
    ErsiliaQualityConfig,
    NeighborConfig,
)
from eosquality.exceptions import IncompatibleArtifactsError
from eosquality.reference.fit_state import FitState, ReferenceReport
from eosquality.reference.metadata import ColumnCharacteristics, FitMetadata
from eosquality.schema.models import ColumnSpec, Schema
from eosquality.utils.logging import logger


def load(path: str | pathlib.Path) -> FitState:
    """Load a FitState from a folder produced by :func:`save`.

    Parameters
    ----------
    path:
        Directory written by :func:`save`.

    Returns
    -------
    FitState
    """
    folder = pathlib.Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"No artifacts folder found at: {folder}")
    if not folder.is_dir():
        raise ValueError(
            f"Expected a directory, got a file: {folder}. "
            "Artifacts are now stored as a folder — pass the folder path."
        )

    logger.debug(f"Loading artifacts from {folder}")

    # JSON artifacts
    with open(folder / "config.json") as f:
        raw_config = json.load(f)
    vector_index_path = raw_config.pop("vector_index_path")
    config = _config_from_dict(raw_config)
    logger.debug("  config.json")

    with open(folder / "schema.json") as f:
        schema = _schema_from_dict(json.load(f))
    logger.debug(f"  schema.json | {len(schema.columns)} columns")

    with open(folder / "reference_report.json") as f:
        reference_report = _reference_report_from_dict(json.load(f))
    logger.debug(
        f"  reference_report.json | quality={reference_report.reference_quality:.4f}"
    )

    with open(folder / "metadata.json") as f:
        metadata = _metadata_from_dict(json.load(f))
    logger.debug(
        f"  metadata.json | {metadata.n_samples:,} samples · {metadata.n_features} features"
    )

    _check_artifacts_compatibility(metadata)

    # Joblib artifacts
    reference_ids = joblib.load(folder / "reference_ids.joblib")
    logger.debug(f"  reference_ids.joblib | {len(reference_ids):,} ids")
    scalers = joblib.load(folder / "scalers.joblib")
    logger.debug(f"  scalers.joblib | {len(scalers)} columns")

    # Numpy arrays
    reference_repr = np.load(folder / "reference_repr.npy")
    logger.debug(f"  reference_repr.npy | shape={reference_repr.shape}")
    reference_knn_distances = np.load(folder / "reference_knn_distances.npy")
    logger.debug(f"  reference_knn_distances.npy | shape={reference_knn_distances.shape}")
    reference_knn_indices = np.load(folder / "reference_knn_indices.npy")
    logger.debug(f"  reference_knn_indices.npy | shape={reference_knn_indices.shape}")

    return FitState(
        config=config,
        schema=schema,
        preprocess_state={
            "schema": schema,
            "scalers": scalers,
            "strategy": config.distance.numeric_scaler,
        },
        reference_ids=reference_ids,
        reference_repr=reference_repr,
        reference_knn_distances=reference_knn_distances,
        reference_knn_indices=reference_knn_indices,
        reference_report=reference_report,
        metadata=metadata,
        vector_index_path=vector_index_path,
    )


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------


def _check_artifacts_compatibility(metadata: FitMetadata) -> None:
    """Refuse to load artifacts produced against a different reference library.

    Two signals are checked (either mismatch is fatal):

    1. ``metadata.library_id`` must equal the canonical :data:`LIBRARY_ID`
       shipped with this install. Artifacts fit against a custom
       ``--vector-index`` whose ``library_name`` differs will be rejected here.
    2. The major version of ``metadata.eosquality_version`` must equal the
       major of the currently installed package. Redundant with (1) for
       normal releases, but catches side-loaded artifacts whose library_id
       somehow matches but were fit under a different major.
    """
    if metadata.library_id != LIBRARY_ID:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit against reference library "
            f"{metadata.library_id!r} but this install ships {LIBRARY_ID!r}. "
            "Major version mismatch — install a compatible eosquality release "
            "or refit against the current library."
        )

    try:
        current = importlib.metadata.version("eosquality")
    except importlib.metadata.PackageNotFoundError:
        return  # source checkout without install; skip the redundant check
    try:
        saved_major = Version(metadata.eosquality_version).major
        current_major = Version(current).major
    except Exception:
        return  # non-PEP440 version string; library_id check above is authoritative
    if saved_major != current_major:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit with eosquality {metadata.eosquality_version} "
            f"(major={saved_major}) but this install is {current} "
            f"(major={current_major}). Reference library is pinned to major — "
            "install a matching eosquality release or refit."
        )


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------


def _config_from_dict(d: dict) -> ErsiliaQualityConfig:
    return ErsiliaQualityConfig(
        distance=DistanceConfig(**d["distance"]),
        neighbors=NeighborConfig(**d["neighbors"]),
        core=CoreConfig(**d["core"]),
        bootstrap=BootstrapConfig(**d["bootstrap"]),
        aggregation=AggregationConfig(**d["aggregation"]),
    )


def _schema_from_dict(d: dict) -> Schema:
    return Schema(
        columns=[ColumnSpec(**c) for c in d["columns"]]
    )


def _reference_report_from_dict(d: dict) -> ReferenceReport:
    return ReferenceReport(**d)


def _metadata_from_dict(d: dict) -> FitMetadata:
    characteristics = {
        col: ColumnCharacteristics(**v)
        for col, v in d.get("column_characteristics", {}).items()
    }
    return FitMetadata(
        eos_id=d["eos_id"],
        version=d["version"],
        n_samples=d["n_samples"],
        n_features=d["n_features"],
        columns=d["columns"],
        column_stats=d["column_stats"],
        missing_counts=d["missing_counts"],
        fit_timestamp=d["fit_timestamp"],
        eosquality_version=d["eosquality_version"],
        column_characteristics=characteristics,
        library_id=d.get("library_id", ""),
    )
