"""Build a SharedFitState from a raw reference DataFrame."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from eosquality.shared.metadata import compute_metadata
from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import infer_schema
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


def fit_shared(
    reference: pd.DataFrame,
    eos_id: str,
    version: str,
    *,
    library_id: str = "",
) -> tuple[SharedFitState, np.ndarray]:
    """Compute the shared fit state from a raw reference DataFrame.

    Returns ``(state, ref_repr)`` — the second value is the scaled
    reference array (``np.ndarray`` of shape ``(n_ref, n_features)``),
    so that downstream index-aware fits don't have to call eosframes a
    second time.

    Records the wall-clock duration of this call on
    ``state.metadata.fit_duration_seconds`` for diagnostics.
    """
    t0 = time.perf_counter()
    schema = infer_schema(reference)
    metadata = compute_metadata(reference, eos_id=eos_id, version=version)
    metadata.library_id = library_id

    logger.reference_table(
        n_samples=len(reference),
        n_features=len(schema.columns),
        column_names=schema.column_names,
    )

    pipeline = PreprocessPipeline(schema=schema)
    ref_repr = pipeline.fit_transform(reference)
    pipeline_state = pipeline.get_state()

    metadata.fit_duration_seconds = float(time.perf_counter() - t0)

    state = SharedFitState(
        schema=schema,
        scaler_params=pipeline_state["scaler_params"],
        binary_class_freq=pipeline_state["binary_class_freq"],
        metadata=metadata,
        reference_ids=list(reference.index),
    )
    return state, ref_repr
