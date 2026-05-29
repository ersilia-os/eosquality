"""Build a SharedFitState from a raw reference DataFrame."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import infer_schema
from eosquality.shared.feature_selection import select_features_by_correlation
from eosquality.shared.metadata import compute_metadata
from eosquality.shared.splitter import Splitter
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


DEFAULT_MAX_FEATURES = 10


def fit_shared(
    reference: pd.DataFrame,
    eos_id: str,
    version: str,
    *,
    library_id: str = "",
    max_features: int | None = DEFAULT_MAX_FEATURES,
) -> tuple[SharedFitState, np.ndarray]:
    """Compute the shared fit state from a raw reference DataFrame.

    Returns ``(state, ref_repr)`` — the second value is the scaled
    reference array (``np.ndarray`` of shape ``(n_ref, n_selected)``).
    When ``max_features`` triggers a reduction, ``ref_repr`` is already
    projected onto ``state.selected_columns``, so downstream index-aware
    fits and per-score fits see the reduced view directly.

    Parameters
    ----------
    max_features:
        Cap on the number of features kept after correlation-cluster
        medoid selection. ``None`` disables the step. When the reference
        has at most ``max_features`` columns, all columns are kept.

    Records the wall-clock duration of this call on
    ``state.metadata.fit_duration_seconds`` for diagnostics.
    """
    t0 = time.perf_counter()
    logger.info(
        f"fit_shared | inferring schema | {len(reference):,} rows × "
        f"{reference.shape[1]} columns"
    )
    schema = infer_schema(reference)
    metadata = compute_metadata(reference, eos_id=eos_id, version=version)
    metadata.library_id = library_id

    logger.reference_table(
        n_samples=len(reference),
        n_features=len(schema.columns),
        column_names=schema.column_names,
    )

    t_pipe = time.perf_counter()
    logger.info(
        f"fit_shared | eosframes fit_transform | {len(reference):,} rows × "
        f"{len(schema.columns)} numeric columns"
    )
    pipeline = PreprocessPipeline(schema=schema)
    ref_repr_full = pipeline.fit_transform(reference)
    pipeline_state = pipeline.get_state()
    logger.info(f"fit_shared | eosframes done | {time.perf_counter() - t_pipe:.1f}s")

    t_fs = time.perf_counter()
    selected_columns = select_features_by_correlation(
        ref_repr_full, schema.column_names, max_features
    )
    if len(selected_columns) < len(schema.column_names):
        name_to_idx = {n: i for i, n in enumerate(schema.column_names)}
        selected_idx = np.asarray(
            [name_to_idx[c] for c in selected_columns], dtype=np.int64
        )
        ref_repr = ref_repr_full[:, selected_idx]
        logger.info(
            f"fit_shared | feature selection | {len(schema.column_names)} → "
            f"{len(selected_columns)} (max_features={max_features}) | "
            f"{time.perf_counter() - t_fs:.1f}s"
        )
    else:
        ref_repr = ref_repr_full
        logger.info(
            f"fit_shared | feature selection skipped "
            f"(n_features={len(schema.column_names)} ≤ max_features={max_features})"
        )

    splits = Splitter().split(len(reference))

    metadata.fit_duration_seconds = float(time.perf_counter() - t0)
    logger.info(f"fit_shared | done | total {metadata.fit_duration_seconds:.1f}s")

    state = SharedFitState(
        schema=schema,
        scaler_params=pipeline_state["scaler_params"],
        binary_class_freq=pipeline_state["binary_class_freq"],
        metadata=metadata,
        reference_ids=list(reference.index),
        splits=splits,
        ref_repr=ref_repr,
        selected_columns=selected_columns,
    )
    return state, ref_repr
