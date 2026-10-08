import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.scores._helpers import _cdf_score
from eosquality.scores.typicality import (
    fit_typicality_luts,
    lookup_percentiles,
    percentile_luts,
)


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        library=library,
        max_features=4,
        exclude=["ref_extremity", "ref_match"],
    )


def test_each_column_is_uniform_on_the_reference(fitted, reference):
    """The per-column percentile averages 0.5 on the reference, column by column."""
    result = fitted.typicality.run(reference)
    assert list(result.per_feature_pct.columns) == fitted.shared_.selected_columns
    np.testing.assert_allclose(result.per_feature_pct.mean(), 0.5, atol=0.02)


def test_whole_model_percentile_is_uniform_on_the_reference(fitted, reference):
    score = fitted.typicality.run(reference).score
    assert score.name == "ref_typicality_pct"
    assert score.mean() == pytest.approx(0.5, abs=0.01)
    assert fitted.typicality.anchor_ == pytest.approx(0.5, abs=0.01)


def test_raw_is_the_q66_of_count_over_max(fitted, reference):
    result = fitted.typicality.run(reference)
    expected = np.nanquantile(result.per_feature.to_numpy(), 0.66, axis=1)
    np.testing.assert_allclose(result.score_raw.to_numpy(), expected)


def test_percentile_table_is_the_mid_rank_of_the_density():
    """Checked against the mid-rank CDF of the per-value densities."""
    rng = np.random.default_rng(0)
    scaled = np.column_stack(
        [
            np.clip(rng.normal(0, 0.3, 500), -1, 1),  # smooth
            rng.integers(0, 2, 500).astype(float),  # binary {0, 1}
            np.zeros(500),  # constant
        ]
    )
    luts = fit_typicality_luts(scaled)
    pct = lookup_percentiles(scaled, percentile_luts(luts))
    for j in range(scaled.shape[1]):
        density = luts[:, j].max()
        levels = np.round(scaled[:, j] * 127).astype(int) + 128
        per_value = luts[levels, j] / density
        expected = _cdf_score(per_value, np.sort(per_value))
        np.testing.assert_allclose(pct[:, j], expected)


def test_a_common_level_scores_above_a_rare_one():
    scaled = np.array([[0.0]] * 90 + [[0.5]] * 9 + [[-1.0]])
    pct = lookup_percentiles(scaled, percentile_luts(fit_typicality_luts(scaled)))
    assert pct[0, 0] > pct[90, 0] > pct[99, 0]


def test_unseen_levels_get_the_floor_and_nan_stays_nan():
    scaled = np.array([[0.0]] * 10)
    pct = lookup_percentiles(
        np.array([[0.9], [np.nan]]), percentile_luts(fit_typicality_luts(scaled))
    )
    assert pct[0, 0] == pytest.approx(0.5 / 10)
    assert np.isnan(pct[1, 0])


def test_reference_details_has_both_columns_per_feature(fitted, query):
    details = fitted.run(query).reference_details
    assert len(details) == len(query)
    cols = fitted.shared_.selected_columns
    assert list(details.columns) == [
        "key",
        "input",
        *[f"{c}_{kind}" for c in cols for kind in ("typicality_raw", "typicality_pct")],
    ]
    first = cols[0]
    np.testing.assert_allclose(
        details[f"{first}_typicality_raw"],
        fitted.typicality.run(query).per_feature[first],
    )


def test_save_load_rebuilds_the_percentile_tables(fitted, query, tmp_path):
    fitted.save(tmp_path / "art")
    loaded = ErsiliaQuality.load(tmp_path / "art")
    np.testing.assert_array_equal(
        loaded.typicality.pct_luts_, fitted.typicality.pct_luts_
    )
    before, after = fitted.run(query), loaded.run(query)
    pd.testing.assert_frame_equal(before.scores, after.scores)
    pd.testing.assert_frame_equal(before.reference_details, after.reference_details)


def test_a_component_can_be_fitted_saved_and_loaded_on_its_own(
    reference, query, tmp_path
):
    """The documented standalone use: shared state is fitted, then saved alongside."""
    from eosquality import Typicality

    alone = Typicality().fit(reference, eos_id="eos0aaa", version="v1")
    alone.save(tmp_path / "art")
    assert (tmp_path / "art" / "shared" / "schema.json").is_file()
    assert (tmp_path / "art" / "typicality" / "count_luts.npy").is_file()
    loaded = Typicality.load(tmp_path / "art")
    pd.testing.assert_frame_equal(
        alone.run(query).per_feature_pct, loaded.run(query).per_feature_pct
    )
    assert loaded.anchor_ == alone.anchor_


def test_loading_a_component_that_was_not_saved_is_a_clear_error(tmp_path):
    from eosquality import Typicality

    with pytest.raises(FileNotFoundError, match="typicality artifacts"):
        Typicality.load(tmp_path)
