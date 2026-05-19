"""FitMetadata: provenance and dataset statistics captured during fit()."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eosquality.schema.infer import ERSILIA_METADATA_COLUMNS
from eosquality.utils.logging import logger


@dataclass
class ColumnCharacteristics:
    """Detected characteristics of a single numeric column."""

    kind: str  # "binary" | "proportion" | "count" | "continuous"
    sparsity: float  # fraction of exact zeros (NaN not counted as zero)
    missing_fraction: float  # fraction of NaN values


def _detect_kind(series: pd.Series) -> str:
    """Detect the kind of a numeric column from its non-null values.

    Priority (most specific first):
      binary     → all non-null values in {0.0, 1.0}
      count      → all non-null values >= 0 and integer-valued
      continuous → default (includes proportions in [0, 1])
    """
    vals = series.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return "continuous"

    unique = set(np.unique(vals))
    if unique <= {0.0, 1.0}:
        return "binary"

    if (vals >= 0.0).all() and (np.abs(vals % 1) < 1e-9).all():
        return "count"

    return "continuous"


def compute_column_characteristics(series: pd.Series) -> ColumnCharacteristics:
    """Compute ColumnCharacteristics for a single numeric column."""
    n = len(series)
    missing_fraction = float(series.isna().sum() / n) if n > 0 else 0.0
    sparsity = float((series == 0).sum() / n) if n > 0 else 0.0
    kind = _detect_kind(series)
    return ColumnCharacteristics(
        kind=kind,
        sparsity=sparsity,
        missing_fraction=missing_fraction,
    )


@dataclass
class FitMetadata:
    """Provenance and dataset statistics for a fitted reference population."""

    eos_id: str  # e.g. "eos4e40"
    version: str  # e.g. "v1"
    n_samples: int  # number of rows in reference
    n_features: int  # number of feature columns
    columns: list[str]  # column names
    column_stats: dict[str, dict[str, float]]  # raw stats per column
    missing_counts: dict[str, int]  # NaN count per column
    fit_timestamp: str  # ISO 8601 UTC
    eosquality_version: str  # package version
    column_characteristics: dict[str, ColumnCharacteristics]  # per-column kind/sparsity
    library_id: str = ""  # e.g. "ersilia_reference_library_v1"
    fit_duration_seconds: float = 0.0  # wall time spent in fit_shared


def compute_metadata(
    df: pd.DataFrame,
    eos_id: str,
    version: str,
) -> FitMetadata:
    """Compute FitMetadata from a raw (unscaled) reference DataFrame.

    Parameters
    ----------
    df:
        The reference DataFrame passed to fit().
    eos_id:
        Validated EOS model identifier (e.g. "eos4e40").
    version:
        Validated dataset version string (e.g. "v1").

    Returns
    -------
    FitMetadata
    """
    # Only report stats for numeric feature columns (skip key/input metadata cols)
    columns = [
        c
        for c in df.columns
        if c not in ERSILIA_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Per-column descriptive stats from raw data
    desc = df.describe()
    column_stats: dict[str, dict[str, float]] = {}
    for col in columns:
        if col in desc.columns:
            col_desc = desc[col]
            column_stats[col] = {
                "mean": float(col_desc.get("mean", float("nan"))),
                "std": float(col_desc.get("std", float("nan"))),
                "min": float(col_desc.get("min", float("nan"))),
                "max": float(col_desc.get("max", float("nan"))),
                "median": float(df[col].median()),
            }
        else:
            column_stats[col] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
            }

    missing_counts = {col: int(df[col].isna().sum()) for col in columns}

    column_characteristics = {
        col: compute_column_characteristics(df[col]) for col in columns
    }

    logger.debug(f"Column characteristics | {len(columns)} columns")
    for col, chars in column_characteristics.items():
        logger.debug(
            f"  {col}: kind={chars.kind}"
            f" | sparsity={chars.sparsity:.3f}"
            f" | missing={chars.missing_fraction:.3f}"
        )

    try:
        eq_version = importlib.metadata.version("eosquality")
    except importlib.metadata.PackageNotFoundError:
        eq_version = "unknown"

    fit_timestamp = datetime.now(tz=timezone.utc).isoformat()

    return FitMetadata(
        eos_id=eos_id,
        version=version,
        n_samples=len(df),
        n_features=len(columns),
        columns=columns,
        column_stats=column_stats,
        missing_counts=missing_counts,
        fit_timestamp=fit_timestamp,
        eosquality_version=eq_version,
        column_characteristics=column_characteristics,
    )
