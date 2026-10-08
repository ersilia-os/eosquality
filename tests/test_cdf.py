import numpy as np
import pytest

from eosquality.scores._helpers import _cdf_score, _nan_aggregate


def test_reference_scores_average_one_half_with_ties():
    ref = np.sort(np.repeat([0.1, 0.2, 0.3], [700, 200, 100]))
    assert _cdf_score(ref, ref).mean() == pytest.approx(0.5)


def test_mid_rank_places_ties_in_middle_of_block():
    ref = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    assert _cdf_score(np.array([2.0]), ref)[0] == pytest.approx(0.5)


def test_range_and_direction():
    ref = np.linspace(0, 1, 101)
    out = _cdf_score(np.array([-1.0, 0.5, 2.0]), ref)
    eps = 1 / (2 * ref.size)
    assert out.tolist() == pytest.approx([eps, 0.5, 1.0])  # higher raw, higher score


def test_nan_passes_through():
    out = _cdf_score(
        np.array([np.nan, 0.5]),
        np.linspace(0, 1, 11),
    )
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        _cdf_score(np.array([0.5]), np.array([]))


def test_nan_aggregate_ignores_missing_features():
    per_feature = np.array([[0.2, np.nan, 0.2], [np.nan, np.nan, np.nan]])
    agg = _nan_aggregate(per_feature)
    assert agg[0] == pytest.approx(0.2)
    assert np.isnan(agg[1])
