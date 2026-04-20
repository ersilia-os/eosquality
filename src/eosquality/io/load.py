"""Deserialize a FitState from a folder of artifact files."""

import json
import pathlib

import joblib
import numpy as np

from eosquality.config import (
    AggregationConfig,
    BootstrapConfig,
    CoreConfig,
    DistanceConfig,
    ErsiliaQualityConfig,
    NeighborConfig,
)
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
    )
