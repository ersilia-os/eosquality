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

    kind: str  # "binary" | "count" | "continuous"
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
    """Compute ColumnCharacteristics for a single numeric column.

    Parameters
    ----------
    series : pandas.Series
        One numeric column.

    Returns
    -------
    ColumnCharacteristics
        Kind, sparsity and missing fraction.
    """
    n = len(series)
    missing_fraction = float(series.isna().sum() / n) if n > 0 else 0.0
    sparsity = float((series == 0).sum() / n) if n > 0 else 0.0
    kind = _detect_kind(series)
    return ColumnCharacteristics(
        kind=kind,
        sparsity=sparsity,
        missing_fraction=missing_fraction,
    )


# On-disk artifact format. Bump whenever saved files change meaning or
# layout so older artifacts fail at load with a clear "refit" message
# instead of producing silently different scores.
#   2 — mid-rank CDF calibration, NaN-ignoring aggregates, merged
#       consistency FP bins, signal CDF as .npy, custom index paths.
#   3 — support calibrated per fingerprint-size bin (reverted in 4).
#   4 — support = nearest-analogue similarity, one reference CDF;
#       support_log output column.
#   5 — artifacts split into reference_mode/ and training_mode/.
#   6 — extremity: per-column reference tables (column_tables.npz); the
#       calibrated score is the CDF of the Q66 of per-column percentiles.
#   7 — typicality: the calibrated score is built the same way, from
#       per-column percentiles of the density (derived from the count LUTs).
#   8 — support, consistency and signal removed (and with them the kNN state,
#       the 80/10/10 splits and the saved scaled reference); ref_match and
#       ref_scaffold added; the metadata names the library by library_path.
ARTIFACT_FORMAT_VERSION = 8


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
    library_id: str = ""  # e.g. "ersilia_reference_library_v0"
    fit_duration_seconds: float = 0.0  # wall time spent in fit_shared
    # Absolute path of a non-canonical library folder; "" for the canonical
    # library, which is resolved by library_id at run time instead.
    library_path: str = ""
    format_version: int = ARTIFACT_FORMAT_VERSION


def compute_metadata(
    df: pd.DataFrame,
    eos_id: str,
    version: str,
) -> FitMetadata:
    """Compute FitMetadata from a raw (unscaled) reference DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The reference passed to ``fit``; only numeric output columns are
        described (``key`` / ``input`` are skipped).
    eos_id : str
        Validated EOS model identifier, e.g. ``"eos4e40"``.
    version : str
        Validated dataset version, e.g. ``"v1"``.

    Returns
    -------
    FitMetadata
        Provenance plus per-column stats, missing counts and characteristics.
    """
    columns = [
        c
        for c in df.columns
        if c not in ERSILIA_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    characteristics = {col: compute_column_characteristics(df[col]) for col in columns}
    for col, chars in characteristics.items():
        logger.debug(
            f"  {col}: kind={chars.kind} | sparsity={chars.sparsity:.3f} | "
            f"missing={chars.missing_fraction:.3f}"
        )
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
        column_stats={col: _column_stats(df[col]) for col in columns},
        missing_counts={col: int(df[col].isna().sum()) for col in columns},
        fit_timestamp=datetime.now(tz=timezone.utc).isoformat(),
        eosquality_version=eq_version,
        column_characteristics=characteristics,
    )


def _column_stats(series: pd.Series) -> dict[str, float]:
    """Mean, std, min, max and median of a numeric column (NaN skipped)."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }
