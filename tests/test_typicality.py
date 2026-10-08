import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.scores._helpers import _cdf_score
from eosquality.scores.typicality import (
    compute_typicality,
    density_histograms,
    fit_typicality_luts,
    lookup_percentiles,
    percentile_tables,
)


def percentiles(reference_scaled, query_scaled=None):
    """Per-feature typicality percentiles of ``query_scaled`` (default: the reference)."""
    luts = fit_typicality_luts(reference_scaled)
    tables = percentile_tables(density_histograms(reference_scaled, luts))
    query = reference_scaled if query_scaled is None else query_scaled
    return lookup_percentiles(compute_typicality(query, luts)[0], tables)


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
    pct = percentiles(scaled)
    density = compute_typicality(scaled, fit_typicality_luts(scaled))[0]
    for j in range(scaled.shape[1]):
        expected = _cdf_score(density[:, j], np.sort(density[:, j]))
        # Binned: exact for ties (binary, constant), within a bin for smooth values.
        np.testing.assert_allclose(pct[:, j], expected, atol=0.02)


def test_percentile_varies_continuously_with_the_value():
    """A smooth column gives far more than the 255 values one level each would."""
    scaled = np.random.default_rng(1).normal(0, 0.2, (20000, 1))
    assert len(np.unique(percentiles(scaled))) > 1000
    grid = np.linspace(0.0, 0.5, 400)[:, None]  # 0.16 of a level per step
    pct = percentiles(scaled, grid)[:, 0]
    assert np.abs(np.diff(pct)).max() < 0.1  # no jump of a whole level's worth


def test_interpolation_is_exact_on_a_level():
    scaled = np.array([[0.0]] * 80 + [[1 / 127]] * 20)
    luts = fit_typicality_luts(scaled)
    per_feature = compute_typicality(np.array([[0.0], [1 / 127], [0.5 / 127]]), luts)[0]
    np.testing.assert_allclose(per_feature[:, 0], [1.0, 0.25, 0.625])


def test_a_common_level_scores_above_a_rare_one():
    scaled = np.array([[0.0]] * 90 + [[0.5]] * 9 + [[-1.0]])
    pct = percentiles(scaled)
    assert pct[0, 0] > pct[90, 0] > pct[99, 0]


def test_unseen_levels_get_the_floor_and_nan_stays_nan():
    scaled = np.array([[0.0]] * 10)
    pct = percentiles(scaled, np.array([[0.9], [np.nan]]))
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
        loaded.typicality.pct_tables_, fitted.typicality.pct_tables_
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
    assert (tmp_path / "art" / "typicality" / "density_hist.npy").is_file()
    loaded = Typicality.load(tmp_path / "art")
    pd.testing.assert_frame_equal(
        alone.run(query).per_feature_pct, loaded.run(query).per_feature_pct
    )
    assert loaded.anchor_ == alone.anchor_


def test_loading_a_component_that_was_not_saved_is_a_clear_error(tmp_path):
    from eosquality import Typicality

    with pytest.raises(FileNotFoundError, match="typicality artifacts"):
        Typicality.load(tmp_path)
