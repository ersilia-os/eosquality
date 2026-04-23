"""Tests for per-column and aggregate typicality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.preprocess.pipeline import PreprocessPipeline
from eosquality.scoring.typicality import compute_typicality
from eosquality.schema.models import ColumnSpec, Schema

EOS_ID = "eos4e40"
VERSION = "v1"


# ---------------------------------------------------------------------------
# Direct unit tests on compute_typicality
# ---------------------------------------------------------------------------


def _make_scaler_continuous(ref_values: np.ndarray) -> dict:
    """Build a scaler dict mirroring what PreprocessPipeline would produce."""
    from eosquality.preprocess.pipeline import _fit_type_aware
    params, _ = _fit_type_aware(ref_values, kind="continuous")
    return params


def _levels() -> np.ndarray:
    return np.linspace(0.0, 1.0, 101)


def test_typicality_at_reference_median_is_one():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 500)
    scaler = _make_scaler_continuous(ref)
    query = np.array([[float(np.median(ref))]])
    per_feat, agg = compute_typicality(
        raw_values=query,
        scalers={"x": scaler},
        column_names=["x"],
        n_reference=len(ref),
        quantile_levels=_levels(),
    )
    assert per_feat.shape == (1, 1)
    assert per_feat[0, 0] > 0.98  # right at the median → tail prob ≈ 0.5 → typ ≈ 1.0
    assert agg[0] > 0.98


def test_typicality_at_extreme_drops_to_eps_floor():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, 500)
    scaler = _make_scaler_continuous(ref)
    far = float(ref.max() + 100.0)
    query = np.array([[far]])
    _, agg = compute_typicality(
        raw_values=query,
        scalers={"x": scaler},
        column_names=["x"],
        n_reference=len(ref),
        quantile_levels=_levels(),
    )
    eps = 1.0 / (2.0 * len(ref))
    # Beyond the max reference value, np.interp clamps cdf to 1, tail_prob=0,
    # and we floor to eps.
    assert agg[0] == pytest.approx(eps, rel=1e-6)


def test_typicality_constant_column_is_one():
    """Constant columns carry no information — typicality is vacuously 1.0."""
    const_ref = np.zeros(200)
    scaler = _make_scaler_continuous(const_ref)
    assert scaler["is_constant"] is True
    _, agg = compute_typicality(
        raw_values=np.array([[0.0], [100.0], [-50.0]]),
        scalers={"x": scaler},
        column_names=["x"],
        n_reference=len(const_ref),
        quantile_levels=_levels(),
    )
    np.testing.assert_allclose(agg, 1.0)


def test_typicality_binary_imbalanced():
    """90/10 class freq: majority class → typicality 1.0, minority → 0.2."""
    ref = np.array([0.0] * 90 + [1.0] * 10)
    from eosquality.preprocess.pipeline import _fit_binary
    scaler, _ = _fit_binary(ref)

    query = np.array([[0.0], [1.0]])
    per_feat, _ = compute_typicality(
        raw_values=query,
        scalers={"x": scaler},
        column_names=["x"],
        n_reference=len(ref),
        quantile_levels=_levels(),
    )
    assert per_feat[0, 0] == pytest.approx(1.0, abs=1e-6)   # majority: min(1, 2*0.9) = 1
    assert per_feat[1, 0] == pytest.approx(0.2, abs=1e-6)   # minority: 2*0.1


def test_typicality_binary_balanced_both_classes_typical():
    ref = np.array([0.0, 1.0] * 50)
    from eosquality.preprocess.pipeline import _fit_binary
    scaler, _ = _fit_binary(ref)
    query = np.array([[0.0], [1.0]])
    per_feat, _ = compute_typicality(
        raw_values=query,
        scalers={"x": scaler},
        column_names=["x"],
        n_reference=len(ref),
        quantile_levels=_levels(),
    )
    np.testing.assert_allclose(per_feat, 1.0, atol=1e-6)


def test_typicality_aggregate_geomean_penalizes_single_outlier():
    """One wildly atypical feature should drag the aggregate well below 1.0,
    even if the other columns are perfectly typical."""
    rng = np.random.default_rng(2)
    ref_a = rng.normal(0.0, 1.0, 300)
    ref_b = rng.normal(10.0, 2.0, 300)
    scalers = {"a": _make_scaler_continuous(ref_a), "b": _make_scaler_continuous(ref_b)}
    query = np.array([[float(np.median(ref_a)), float(ref_b.max()) + 100.0]])
    per_feat, agg = compute_typicality(
        raw_values=query,
        scalers=scalers,
        column_names=["a", "b"],
        n_reference=300,
        quantile_levels=_levels(),
    )
    # per_feat[0, 0] ≈ 1 (at median), per_feat[0, 1] at eps floor.
    assert per_feat[0, 0] > 0.95
    assert per_feat[0, 1] < 0.01
    # geomean pulls way down
    assert agg[0] < 0.1


# ---------------------------------------------------------------------------
# Integration through fit → run
# ---------------------------------------------------------------------------


def test_run_exposes_typicality_per_feature(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    result = eq.run(query_df_vi)
    per_feature = result.typicality_per_feature
    assert isinstance(per_feature, pd.DataFrame)
    assert per_feature.shape == (len(query_df_vi), 3)
    assert list(per_feature.columns) == ["feat_a", "feat_b", "feat_c"]
    assert (per_feature.values >= 0.0).all()
    assert (per_feature.values <= 1.0).all()


def test_run_typicality_score_in_scores_df(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    result = eq.run(query_df_vi)
    assert "typicality_score" in result.scores.columns
    vals = result.scores["typicality_score"].to_numpy()
    assert (vals >= 0.0).all()
    assert (vals <= 1.0).all()


def test_reference_typicality_is_sensible_baseline(
    reference_df_vi, vector_index_dir,
):
    """The reference's aggregate typicality is a calibration baseline.

    Per-feature typicality averages ~0.5 on a random reference sample
    (E[2·min(U, 1−U)] = 0.5 for uniform U), and the geomean across features
    pulls that lower. We only assert the value is in (0, 1] — the precise
    magnitude depends on n_features and the empirical distribution shape.
    """
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    ref_typ = eq.reference_report_.reference_typicality
    assert 0.0 < ref_typ <= 1.0


def test_typicality_roundtrip_via_save_load(
    reference_df_vi, query_df_vi, vector_index_dir, tmp_path,
):
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    before = eq.run(query_df_vi)
    folder = tmp_path / "ta"
    eq.save(folder)
    eq2 = ErsiliaQuality.load(folder)
    after = eq2.run(query_df_vi)
    np.testing.assert_allclose(
        before.scores["typicality_score"].to_numpy(),
        after.scores["typicality_score"].to_numpy(),
    )
    pd.testing.assert_frame_equal(
        before.typicality_per_feature,
        after.typicality_per_feature,
    )
