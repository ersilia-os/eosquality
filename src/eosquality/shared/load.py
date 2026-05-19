"""Read a SharedFitState from <root>/shared/."""

from __future__ import annotations

import json
import pathlib

from eosquality.shared.metadata import ColumnCharacteristics, FitMetadata
from eosquality.schema.models import ColumnSpec, Schema
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


SUBFOLDER = "shared"


def load_shared(root: str | pathlib.Path) -> SharedFitState:
    """Read the SharedFitState from ``<root>/shared/``."""
    folder = pathlib.Path(root) / SUBFOLDER
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Expected shared fit state at {folder}, but the folder does not exist."
        )

    with open(folder / "schema.json") as f:
        schema = Schema(columns=[ColumnSpec(**c) for c in json.load(f)["columns"]])
    with open(folder / "scaler.json") as f:
        scaler_params = json.load(f)
    with open(folder / "binary_class_freq.json") as f:
        binary_class_freq = json.load(f)
    with open(folder / "metadata.json") as f:
        metadata = _metadata_from_dict(json.load(f))
    with open(folder / "reference_ids.json") as f:
        reference_ids = json.load(f)

    logger.debug(f"  shared/ | {len(schema.columns)} columns")
    return SharedFitState(
        schema=schema,
        scaler_params=scaler_params,
        binary_class_freq=binary_class_freq,
        metadata=metadata,
        reference_ids=reference_ids,
    )


def _metadata_from_dict(d: dict) -> FitMetadata:
    """Reconstruct a :class:`FitMetadata` instance from its JSON payload."""
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
        fit_duration_seconds=float(d.get("fit_duration_seconds", 0.0)),
    )
