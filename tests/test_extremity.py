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
    )
    np.testing.assert_allclose(result.per_feature_pct[name], expected)
    assert table.dtype == np.float32 and (np.diff(table) >= 0).all()


def test_sign_is_ignored():
    """Mirrored scaled values are equally extreme; the rails clip at 1."""
    from eosquality.scores.extremity import per_feature_extremity

    per_feature = per_feature_extremity(np.array([[0.3, -0.7, 2.0], [-0.3, 0.7, -2.0]]))
    np.testing.assert_allclose(per_feature[0], per_feature[1])
    np.testing.assert_allclose(per_feature[0], [0.3, 0.7, 1.0])


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
