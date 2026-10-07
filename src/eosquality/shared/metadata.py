"""FitMetadata: provenance captured during fit()."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from eosquality.schema.infer import ERSILIA_METADATA_COLUMNS

# On-disk artifact format. Bump whenever saved files change meaning or
# layout so older artifacts fail at load with a clear "refit" message
# instead of producing silently different scores. History: git log.
ARTIFACT_FORMAT_VERSION = 8


@dataclass
class FitMetadata:
    """Provenance of a fitted reference population."""

    eos_id: str  # e.g. "eos4e40"
    version: str  # e.g. "v1"
    n_samples: int  # number of rows in reference
    n_features: int  # number of feature columns
    columns: list[str]  # column names
    fit_timestamp: str  # ISO 8601 UTC
    eosquality_version: str  # package version
    library_id: str = ""  # e.g. "ersilia_reference_library_v0"
    fit_duration_seconds: float = 0.0  # wall time spent in fit_shared
    # Absolute path of a non-canonical library folder; "" for the canonical
    # library, which is resolved by library_id at run time instead.
    library_path: str = ""
    format_version: int = ARTIFACT_FORMAT_VERSION


def compute_metadata(df: pd.DataFrame, eos_id: str, version: str) -> FitMetadata:
    """Compute FitMetadata from a raw (unscaled) reference DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The reference passed to ``fit``; only its numeric output columns are
        counted (``key`` / ``input`` are skipped).
    eos_id : str
        Validated EOS model identifier, e.g. ``"eos4e40"``.
    version : str
        Validated dataset version, e.g. ``"v1"``.

    Returns
    -------
    FitMetadata
    """
    columns = [
        c
        for c in df.columns
        if c not in ERSILIA_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    try:
        eq_version = importlib.metadata.version("eosquality")
    except importlib.metadata.PackageNotFoundError:
        eq_version = "unknown"
    return FitMetadata(
        eos_id=eos_id,
        version=version,
        n_samples=len(df),
        n_features=len(columns),
        columns=columns,
        fit_timestamp=datetime.now(tz=UTC).isoformat(),
        eosquality_version=eq_version,
    )
