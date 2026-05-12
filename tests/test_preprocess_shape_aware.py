"""Tests for the eosframes-backed preprocessing pipeline contract.

The scaler math itself lives in :mod:`eosframes`; here we only assert on the
thin wrapper's contract: fit_transform shape, transform parity after a
fit, state round-trip via get_state/from_state, and that the eosframes
params dict is the artifact we store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eosquality.preprocess.pipeline import PreprocessPipeline
from eosquality.schema.models import ColumnSpec, Schema


def _schema_for(cols: list[str]) -> Schema:
    return Schema(columns=[ColumnSpec(name=c, kind="numeric") for c in cols])


def test_fit_transform_returns_2d_array_in_unit_box(rng):
    """Output should be the scaled feature array, one row per input row."""
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, n),
            "b": rng.exponential(1.0, n),
            "c": rng.integers(0, 2, n).astype(float),
        }
    )
    pipeline = PreprocessPipeline(schema=_schema_for(["a", "b", "c"]))
    out = pipeline.fit_transform(df)
    assert out.shape == (n, 3)
    # eosframes outputs are bounded inside [-1, 1] per documented per-kind
    # regions; values strictly outside that mean something has broken.
    assert np.nanmin(out) >= -1.0 - 1e-6
    assert np.nanmax(out) <= 1.0 + 1e-6


def test_transform_matches_fit_transform_on_same_data(rng):
    """transform() applied to the fit data should match fit_transform output."""
    n = 150
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, n),
            "b": rng.normal(5.0, 2.0, n),
        }
    )
    pipeline = PreprocessPipeline(schema=_schema_for(["a", "b"]))
    fit_out = pipeline.fit_transform(df)
    transform_out = pipeline.transform(df)
    np.testing.assert_allclose(fit_out, transform_out)


def test_state_roundtrip_preserves_transform(rng):
    """get_state → from_state must reproduce the same transform."""
    n = 100
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, n),
            "b": rng.normal(-3.0, 0.5, n),
        }
    )
    p1 = PreprocessPipeline(schema=_schema_for(["a", "b"]))
    out1 = p1.fit_transform(df)
    p2 = PreprocessPipeline.from_state(p1.get_state())
    out2 = p2.transform(df)
    np.testing.assert_allclose(out1, out2)


def test_state_contains_eosframes_scaler_params(rng):
    """The persisted state must carry an eosframes-style params dict.

    Downstream callers (typicality scorer, save/load) rely on the
    ``method`` / ``columns`` keys to dispatch.
    """
    n = 50
    df = pd.DataFrame({"a": rng.normal(0.0, 1.0, n)})
    pipeline = PreprocessPipeline(schema=_schema_for(["a"]))
    pipeline.fit_transform(df)
    state = pipeline.get_state()
    params = state["scaler_params"]
    assert params["method"] == "robust_typed"
    assert "a" in params["columns"]
    assert "transform" in params["columns"]["a"]
    assert "kind" in params["columns"]["a"]["transform"]


def test_transform_before_fit_raises():
    pipeline = PreprocessPipeline(schema=_schema_for(["a"]))
    try:
        pipeline.transform(pd.DataFrame({"a": [1.0, 2.0]}))
    except RuntimeError:
        return
    raise AssertionError("transform() before fit_transform() should raise.")
