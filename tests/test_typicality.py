"""Tests for per-column and aggregate typicality.

Typicality is now derived from the eosframes int8 quantization:
typicality = 1 - |int8| / 127 per column, kind-aware (constant and binary
contribute 1.0 unconditionally).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.scoring.typicality import compute_typicality

EOS_ID = "eos4e40"
VERSION = "v1"


# ---------------------------------------------------------------------------
# Direct unit tests on compute_typicality
# ---------------------------------------------------------------------------


def _params(kind: str, cols: list[str]) -> dict:
    """Build a minimal eosframes-style params dict for the given kinds."""
    return {
        "method": "robust_typed",
        "columns": {
            c: {"transform": {"kind": kind}, "impute_value": 0.0}
            for c in cols
        },
    }


def test_typicality_at_body_anchor_is_one():
    """Scaled value at 0 (body anchor) → int8 0 → typicality 1.0."""
    scaled = np.array([[0.0]])
    per_feat, agg = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("continuous_centered", ["x"]),
        column_names=["x"],
        n_reference=500,
    )
    assert per_feat[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert agg[0] == pytest.approx(1.0, abs=1e-9)


def test_typicality_at_region_edge_drops_to_eps_floor():
    """Scaled value at ±1 (Tukey-fence edge) → |int8|=127 → typicality 0 → eps floor."""
    n_ref = 500
    eps = 1.0 / (2.0 * n_ref)
    scaled = np.array([[1.0], [-1.0]])
    _, agg = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("continuous_centered", ["x"]),
        column_names=["x"],
        n_reference=n_ref,
    )
    assert agg[0] == pytest.approx(eps, rel=1e-6)
    assert agg[1] == pytest.approx(eps, rel=1e-6)


def test_typicality_constant_kind_is_one():
    """Constant kind carries no information — typicality is vacuously 1.0."""
    scaled = np.array([[0.0], [0.5], [-0.5]])
    _, agg = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("constant", ["x"]),
        column_names=["x"],
        n_reference=200,
    )
    np.testing.assert_allclose(agg, 1.0)


def test_typicality_binary_kind_is_one_for_both_classes():
    """Binary kind: eosframes int8 doesn't encode class freq, so typicality
    is 1.0 for both classes by design."""
    scaled = np.array([[0.0], [1.0]])
    per_feat, _ = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("binary", ["x"]),
        column_names=["x"],
        n_reference=200,
    )
    np.testing.assert_allclose(per_feat, 1.0, atol=1e-9)


def test_typicality_nan_maps_to_one():
    """NaN scaled values carry no information → typicality 1.0."""
    scaled = np.array([[np.nan]])
    per_feat, _ = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("continuous_centered", ["x"]),
        column_names=["x"],
        n_reference=200,
    )
    assert per_feat[0, 0] == pytest.approx(1.0, abs=1e-9)


def test_typicality_aggregate_geomean_penalizes_single_outlier():
    """One column at the body anchor and one at the edge → geomean pulled way down."""
    scaled = np.array([[0.0, 1.0]])
    per_feat, agg = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("continuous_centered", ["a", "b"]),
        column_names=["a", "b"],
        n_reference=300,
    )
    assert per_feat[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert per_feat[0, 1] < 0.01
    assert agg[0] < 0.1


def test_typicality_one_sided_kind_treats_zero_as_typical():
    """For one-sided kinds (right-skew, count_zero_mode), the body anchor is
    still 0 in the scaled space; values approaching +1 are atypical."""
    scaled = np.array([[0.0], [1.0]])
    per_feat, _ = compute_typicality(
        scaled_values=scaled,
        scaler_params=_params("continuous_right_skew", ["x"]),
        column_names=["x"],
        n_reference=500,
    )
    assert per_feat[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert per_feat[1, 0] < 0.01


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

    With int8-based typicality the magnitude depends on how the eosframes
    scaler shapes the reference: values near the body anchor (≈ 0) score
    near 1, and Tukey-fence-territory values score near 0. We only assert
    the value is in (0, 1].
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
