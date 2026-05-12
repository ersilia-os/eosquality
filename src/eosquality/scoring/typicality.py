"""Per-column typicality from the eosframes int8-quantized scaled output.

For each query sample and each feature column we compute a typicality in
``[eps, 1]`` from the column's int8-quantized scaled value:

- The eosframes scaler maps every column's raw value into a per-kind region
  inside ``[-1, 1]``. Int8 quantization is the uniform map
  ``int8 = round(scaled · 127)`` clipped to ``[-127, 127]``; ``-128`` is the
  NaN sentinel.
- Typicality treats the body anchor (int8 ≈ 0) as maximally typical and the
  region edges (``|int8| = 127``) as maximally atypical:
  ``typicality = 1 - |int8| / 127``.
- ``constant`` columns carry no information and score 1.0 regardless.
- ``binary`` columns also score 1.0 unconditionally: eosframes' binary
  transform snaps to ``{0, 1}`` and the int8 levels are ``{0, 127}`` for
  both classes, so the int8 magnitude alone can't tell majority from
  minority. If/when class-frequency-aware binary typicality is wanted,
  persist ``class_freq`` per binary column alongside ``scaler_params``.
- The NaN sentinel (``int8 == -128``) maps to typicality 1.0 — NaN carries
  no information and should not drag the geometric mean down.

The aggregate per query is a geometric mean across features, floored at
``eps = 1 / (2 · n_reference)`` so a single off-chart feature can't collapse
the score to zero.
"""

from __future__ import annotations

import numpy as np

_INT8_MAX_VAL = 127
_INT8_NAN_SENTINEL = -128

# Kinds that don't carry typicality information in the int8 scheme. Listed
# explicitly rather than inferred so adding a new eosframes kind raises an
# obvious failure here instead of silently defaulting.
_NO_INFO_KINDS = frozenset({"constant", "binary"})


def compute_typicality(
    scaled_values: np.ndarray,
    scaler_params: dict,
    column_names: list[str],
    n_reference: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature and aggregate typicality for a batch of queries.

    Parameters
    ----------
    scaled_values:
        ``(n_query, n_features)`` float array produced by the eosframes
        scaler (the output of :meth:`PreprocessPipeline.transform`). Column
        order must match ``column_names``.
    scaler_params:
        The dict returned by ``eosframes.fit`` (stored in
        ``preprocess_state["scaler_params"]``). Provides per-column kind
        dispatch.
    column_names:
        Names of the feature columns in order.
    n_reference:
        Number of reference samples, used to compute the eps floor.

    Returns
    -------
    per_feature:
        ``(n_query, n_features)`` typicality scores in ``[eps, 1]``.
    aggregate:
        ``(n_query,)`` aggregated typicality in ``[eps, 1]`` — the geometric
        mean across features.
    """
    n_query = scaled_values.shape[0]
    n_features = scaled_values.shape[1]
    if n_features == 0:
        return np.ones((n_query, 0)), np.ones(n_query)

    eps = 1.0 / (2.0 * max(n_reference, 1))

    columns_params = scaler_params["columns"]
    per_feature = np.empty((n_query, n_features), dtype=np.float64)
    for j, col in enumerate(column_names):
        kind = columns_params[col]["transform"]["kind"]
        per_feature[:, j] = _typicality_for_column(
            scaled_values[:, j], kind=kind, eps=eps
        )

    # All factors >= eps > 0 by construction, so the log is safe.
    aggregate = np.exp(np.log(per_feature).mean(axis=1))
    return per_feature, aggregate


def _typicality_for_column(
    scaled: np.ndarray,
    kind: str,
    eps: float,
) -> np.ndarray:
    """Per-column typicality from the float scaled values.

    Mirrors the eosframes int8 quantization: ``int8 = round(x · 127)`` with
    NaN → sentinel ``-128``; for non-no-info kinds the typicality is
    ``1 - |int8| / 127`` and the sentinel maps back to typicality 1.0
    (NaN carries no information).
    """
    if kind in _NO_INFO_KINDS:
        return np.ones_like(scaled, dtype=np.float64)

    nan_mask = np.isnan(scaled)
    # Quantize non-NaN values to int8 levels, then map back to typicality.
    quantized = np.where(
        nan_mask, 0.0, np.round(scaled * _INT8_MAX_VAL)
    )
    quantized = np.clip(quantized, -_INT8_MAX_VAL, _INT8_MAX_VAL)
    typ = 1.0 - np.abs(quantized) / _INT8_MAX_VAL
    # NaN → 1.0 (no information).
    typ = np.where(nan_mask, 1.0, typ)
    return np.clip(typ, eps, 1.0)
