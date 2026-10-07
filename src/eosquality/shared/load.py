"""Read a SharedFitState from <root>/shared/."""

from __future__ import annotations

import json
import pathlib
from dataclasses import fields

from eosquality.exceptions import ArtifactVersionError
from eosquality.schema.models import ColumnSpec, Schema
from eosquality.shared.metadata import (
    ARTIFACT_FORMAT_VERSION,
    FitMetadata,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger

SUBFOLDER = "shared"


def load_shared(root: str | pathlib.Path) -> SharedFitState:
    """Read the SharedFitState from ``<root>/shared/``.

    Parameters
    ----------
    root : str or pathlib.Path
        Folder that contains ``shared/``.

    Returns
    -------
    SharedFitState
    """
    folder = pathlib.Path(root) / SUBFOLDER
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Expected shared fit state at {folder}, but the folder does not exist."
        )

    with open(folder / "metadata.json") as f:
        metadata = _metadata_from_dict(json.load(f))
    if metadata.format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactVersionError(
            f"Artifacts at {root} use format version {metadata.format_version}; "
            f"this eosquality install reads format {ARTIFACT_FORMAT_VERSION}. "
            "Refit with the current version."
        )
    with open(folder / "schema.json") as f:
        schema = Schema(columns=[ColumnSpec(**c) for c in json.load(f)["columns"]])
    with open(folder / "scaler.json") as f:
        scaler_params = json.load(f)
    with open(folder / "binary_class_freq.json") as f:
        binary_class_freq = json.load(f)
    with open(folder / "selected_columns.json") as f:
        selected_columns = list(json.load(f)["selected_columns"])

    logger.debug(
        f"  shared/ | {len(schema.columns)} columns | selected {len(selected_columns)}"
    )
    return SharedFitState(
        schema=schema,
        scaler_params=scaler_params,
        binary_class_freq=binary_class_freq,
        metadata=metadata,
        selected_columns=selected_columns,
    )


def _metadata_from_dict(d: dict) -> FitMetadata:
    """Reconstruct a :class:`FitMetadata` from its JSON payload (unknown keys are ignored)."""
    known = {f.name for f in fields(FitMetadata)}
    # Artifacts written before format versioning have no field: format 1.
    return FitMetadata(
        **{"format_version": 1, **{k: v for k, v in d.items() if k in known}}
    )
