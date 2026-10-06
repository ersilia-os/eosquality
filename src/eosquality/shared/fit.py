"""Build a SharedFitState from a raw reference DataFrame."""

from __future__ import annotations

import time

import pandas as pd

from eosquality._registry import DEFAULT_MAX_FEATURES
from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import infer_schema
from eosquality.shared.feature_selection import select_features_by_correlation
from eosquality.shared.metadata import compute_metadata
from eosquality.shared.splitter import Splitter
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


def fit_shared(
    reference: pd.DataFrame,
    eos_id: str,
    version: str,
    *,
    library_id: str = "",
    vector_index_path: str = "",
    max_features: int | None = DEFAULT_MAX_FEATURES,
) -> SharedFitState:
    """Compute the shared fit state from a raw reference DataFrame.

    Parameters
    ----------
    reference : pandas.DataFrame
        Predictions on the reference library.
    eos_id, version : str
        Model identifier and dataset version.
    library_id : str, optional
        Identifier of the vector index the reference is aligned with.
    vector_index_path : str, optional
        Absolute path of a non-canonical index (``""`` for the canonical one).
    max_features : int, optional
        Cap on columns kept by correlation-cluster medoid selection; ``None``
        disables it.

    Returns
    -------
    SharedFitState
        Schema, scaler, metadata, splits, selected columns and ``ref_repr``
        (the scaled reference projected onto the selected columns).
    """
    t0 = time.perf_counter()
    schema = infer_schema(reference)
    metadata = compute_metadata(reference, eos_id=eos_id, version=version)
    metadata.library_id = library_id
    metadata.vector_index_path = vector_index_path
    logger.debug(
        f"shared | reference {len(reference):,} rows · {len(schema.columns)} "
        f"columns: {', '.join(schema.column_names[:8])}"
        + ("…" if len(schema.columns) > 8 else "")
    )
    pipeline = PreprocessPipeline(schema=schema)
    ref_repr_full = pipeline.fit_transform(reference)
    pipeline_state = pipeline.get_state()
    selected_columns = select_features_by_correlation(
        ref_repr_full, schema.column_names, max_features
    )
    name_to_idx = {n: i for i, n in enumerate(schema.column_names)}
    ref_repr = (
        ref_repr_full[:, [name_to_idx[c] for c in selected_columns]]
        if len(selected_columns) < len(schema.column_names)
        else ref_repr_full
    )
    metadata.fit_duration_seconds = float(time.perf_counter() - t0)
    logger.info(
        f"fit_shared | {len(reference):,} rows × {len(schema.columns)} columns → "
        f"{len(selected_columns)} selected | {metadata.fit_duration_seconds:.1f}s"
    )
    return SharedFitState(
        schema=schema,
        scaler_params=pipeline_state["scaler_params"],
        binary_class_freq=pipeline_state["binary_class_freq"],
        metadata=metadata,
        reference_ids=list(reference.index),
        splits=Splitter().split(len(reference)),
        ref_repr=ref_repr,
        selected_columns=selected_columns,
    )
