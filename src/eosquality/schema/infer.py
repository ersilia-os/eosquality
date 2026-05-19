"""Infer a Schema from a pandas DataFrame."""

import pandas as pd

from eosquality.exceptions import SchemaError
from eosquality.schema.models import ColumnSpec, Schema
from eosquality.utils.logging import logger

# Standard Ersilia output columns: present in every model output CSV but are
# not numeric features. They are stripped silently during schema inference.
ERSILIA_METADATA_COLUMNS = {"key", "input"}


def infer_schema(df: pd.DataFrame) -> Schema:
    """Infer column types from a DataFrame.

    ``key`` and ``input`` are standard Ersilia metadata columns — they are
    stripped silently with no warning. Any other non-numeric column raises
    :class:`SchemaError`. Raises :class:`SchemaError` also if the DataFrame
    is empty or contains no numeric columns at all.
    """
    if df.empty:
        raise SchemaError("Input DataFrame is empty.")

    columns: list[ColumnSpec] = []
    unexpected_non_numeric: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(ColumnSpec(name=str(col), kind="numeric"))
        elif col in ERSILIA_METADATA_COLUMNS:
            pass  # expected Ersilia metadata column — strip silently
        else:
            unexpected_non_numeric.append(str(col))

    if unexpected_non_numeric:
        raise SchemaError(
            f"Unexpected non-numeric column(s): {unexpected_non_numeric}. "
            "Only 'key' and 'input' are accepted as non-numeric columns "
            "(standard Ersilia output metadata)."
        )

    if not columns:
        raise SchemaError(
            "No numeric columns found. "
            "The DataFrame contains no feature columns (only metadata columns "
            "such as 'key' and 'input' are present)."
        )

    stripped = [c for c in df.columns if c in ERSILIA_METADATA_COLUMNS]
    logger.debug(
        f"Schema inferred | {len(columns)} numeric column(s)"
        + (f" | stripped metadata: {stripped}" if stripped else "")
    )
    return Schema(columns=columns)


def validate_against_schema(df: pd.DataFrame, schema: Schema) -> None:
    """Check that a DataFrame contains all columns that were present at fit time.

    ``key`` and ``input`` are silently ignored. Raises :class:`SchemaError`
    if a fitted numeric column is missing or is no longer numeric.
    """
    missing = [c for c in schema.column_names if c not in df.columns]
    if missing:
        raise SchemaError(f"Columns present at fit time are missing: {missing}")

    non_numeric = [
        c for c in schema.column_names if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise SchemaError(f"Expected numeric columns, got non-numeric: {non_numeric}")
