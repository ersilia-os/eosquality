import numpy as np

from eosquality.scores._helpers import _query_output_distances
from eosquality.scores.extremity import compute_extremity
from eosquality.scores.typicality import compute_typicality, fit_typicality_luts


def test_typicality_ignores_nan_features():
    ref = np.random.default_rng(0).uniform(-1, 1, size=(500, 3))
    luts = fit_typicality_luts(ref)
    q = np.array([[0.1, np.nan, 0.1], [np.nan, np.nan, np.nan]])
    per_feature, agg = compute_typicality(q, luts)
    assert np.isnan(per_feature[0, 1])
    assert np.isfinite(agg[0]) and np.isnan(agg[1])
    # Same aggregate as the row without the missing feature.
    _, agg_two = compute_typicality(q[:1, [0, 2]], luts[:, [0, 2]])
    assert agg[0] == agg_two[0]


def test_typicality_nan_sentinel_does_not_collide():
    ref = np.full((10, 1), -1.0)
    luts = fit_typicality_luts(ref)
    per_feature, _ = compute_typicality(
        np.array([[-1.5]]), luts
    )  # out of range, finite
    assert per_feature[0, 0] == 1.0


def test_extremity_ignores_nan_features():
    per_feature, agg = compute_extremity(np.array([[0.5, np.nan], [np.nan, np.nan]]))
    assert agg[0] == 0.5 and np.isnan(agg[1])


def test_output_distance_uses_finite_features_only():
    q = np.array([[0.0, np.nan]])
    ref = np.array([[1.0, 5.0], [np.nan, np.nan]])
    d = _query_output_distances(q, ref, np.array([[0, 1]]))
    assert d[0, 0] == 1.0 and np.isnan(d[0, 1])
