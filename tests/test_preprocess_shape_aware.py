"""Tests for shape-aware preprocessing: quantile grid, kind dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from eosquality.preprocess.pipeline import (
    _fit_binary,
    _fit_column,
    _fit_proportion,
    _fit_type_aware,
)


def test_quantile_grid_length_is_101():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 500)
    params, _ = _fit_type_aware(values, kind="continuous")
    assert params["quantiles"].shape == (101,)


def test_quantile_grid_captures_bimodal():
    """A bimodal distribution shouldn't have its middle quantiles collapse
    to a single center — the stored grid should span both modes."""
    rng = np.random.default_rng(1)
    mode_a = rng.normal(-5.0, 0.3, 500)
    mode_b = rng.normal(5.0, 0.3, 500)
    values = np.concatenate([mode_a, mode_b])
    params, _ = _fit_type_aware(values, kind="continuous")
    # The lower half of the grid should still sit near -5, upper half near +5,
    # indicating the bimodal structure is preserved.
    assert params["quantiles"][25] < -2.0
    assert params["quantiles"][75] > 2.0


def test_quantile_grid_densifies_toward_right_tail_on_skewed():
    rng = np.random.default_rng(2)
    # exponential + spike at zero: heavy right tail
    values = np.concatenate([np.zeros(400), rng.exponential(1.0, 600)])
    params, _ = _fit_type_aware(values, kind="continuous")
    q = params["quantiles"]
    # Steps between adjacent upper-tail quantiles should be wider than
    # steps between adjacent lower-tail quantiles.
    lower_step = float(q[50] - q[40])
    upper_step = float(q[99] - q[90])
    assert upper_step > lower_step


def test_binary_class_freq_stored_correctly():
    ref = np.array([0.0] * 70 + [1.0] * 30)
    params, out = _fit_binary(ref)
    assert params["kind"] == "binary"
    assert params["class_freq"] == {0.0: 0.7, 1.0: 0.3}
    assert params["quantiles"] is None
    # Pass-through: output equals input.
    np.testing.assert_array_equal(out, ref)


def test_binary_handles_constant_column():
    ref = np.zeros(50)
    params, _ = _fit_binary(ref)
    assert params["is_constant"] is True
    assert params["class_freq"] == {0.0: 1.0, 1.0: 0.0}


def test_proportion_is_pass_through():
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 1.0, 200)
    params, out = _fit_proportion(values)
    assert params["kind"] == "proportion"
    assert params["use_log1p"] is False
    np.testing.assert_allclose(out, values)


def test_kind_dispatcher_routes_correctly():
    rng = np.random.default_rng(4)
    continuous = rng.normal(0.0, 1.0, 50)
    binary = rng.integers(0, 2, 50).astype(float)
    proportion = rng.uniform(0.0, 1.0, 50)

    p_cont, _ = _fit_column(continuous, kind="continuous")
    p_bin, _ = _fit_column(binary, kind="binary")
    p_prop, _ = _fit_column(proportion, kind="proportion")

    assert p_cont["kind"] == "continuous"
    assert p_bin["kind"] == "binary"
    assert p_prop["kind"] == "proportion"

    assert p_cont["quantiles"] is not None
    assert p_bin["quantiles"] is None
    assert p_prop["quantiles"] is not None


def test_scaler_schema_version_stamped():
    rng = np.random.default_rng(5)
    params, _ = _fit_type_aware(rng.normal(0.0, 1.0, 100), kind="continuous")
    assert params["scaler_schema_version"] == 2
