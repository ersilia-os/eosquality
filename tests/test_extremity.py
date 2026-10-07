import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.scores._helpers import _cdf_score


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        library=library,
        max_features=4,
        exclude=["ref_typicality", "ref_match"],
    )


def test_each_column_is_uniform_on_the_reference(fitted, reference):
    """The per-column percentile averages 0.5 on the reference, column by column."""
    result = fitted.extremity.run(reference)
    assert list(result.per_feature_pct.columns) == fitted.shared_.selected_columns
    np.testing.assert_allclose(result.per_feature_pct.mean(), 0.5, atol=0.02)


def test_whole_model_percentile_is_uniform_on_the_reference(fitted, reference):
    score = fitted.extremity.run(reference).score
    assert score.mean() == pytest.approx(0.5, abs=0.01)
    assert fitted.extremity.anchor_ == pytest.approx(0.5, abs=0.01)


def test_raw_is_the_q66_of_the_per_column_values(fitted, query):
    result = fitted.extremity.run(query)
    expected = np.nanquantile(result.per_feature.to_numpy(), 0.66, axis=1)
    np.testing.assert_allclose(result.score_raw, expected)
    finite = result.per_feature.to_numpy()
    finite = finite[np.isfinite(finite)]
    assert (finite >= 0).all() and (finite <= 1).all()


def test_percentile_is_looked_up_on_the_columns_own_table(fitted, query):
    result = fitted.extremity.run(query)
    name = fitted.shared_.selected_columns[0]
    table = fitted.extremity.column_tables_[name]
    expected = _cdf_score(
        result.per_feature[name].to_numpy().astype(np.float32),
        table,
        higher_is_higher=True,
    )
    np.testing.assert_allclose(result.per_feature_pct[name], expected)
    assert table.dtype == np.float32 and (np.diff(table) >= 0).all()


def test_sign_is_ignored(fitted, reference):
    """Mirrored scaled values are equally extreme."""
    from eosquality.scores.extremity import compute_extremity

    values = np.array([[0.3, -0.7], [-0.3, 0.7]])
    per_feature, agg = compute_extremity(values)
    np.testing.assert_allclose(per_feature[0], per_feature[1])
    assert agg[0] == agg[1]


def test_reference_details_has_one_row_per_query(fitted, query):
    result = fitted.run(query)
    details = result.reference_details
    assert len(details) == len(query)
    cols = fitted.shared_.selected_columns
    assert list(details.columns) == [
        "key",
        "input",
        *[f"{c}_{kind}" for c in cols for kind in ("extremity_raw", "extremity_pct")],
    ]
    first = cols[0]
    np.testing.assert_allclose(
        details[f"{first}_extremity_raw"],
        fitted.extremity.run(query).per_feature[first],
    )


def test_save_load_keeps_the_tables(fitted, query, tmp_path):
    fitted.save(tmp_path / "art")
    loaded = ErsiliaQuality.load(tmp_path / "art")
    before, after = fitted.run(query), loaded.run(query)
    pd.testing.assert_frame_equal(before.scores, after.scores)
    pd.testing.assert_frame_equal(before.reference_details, after.reference_details)


def test_no_reference_details_without_typicality_or_extremity(reference, library):
    eq = ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        library=library,
        max_features=4,
        exclude=[
            "ref_typicality",
            "ref_extremity",
        ],
    )
    assert eq.run(reference.head(5)).reference_details is None
