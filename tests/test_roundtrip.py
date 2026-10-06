import json

import numpy as np
import pandas as pd
import pytest

from eosquality import ALL_SCORES, ErsiliaQuality, Support, Typicality
from eosquality.exceptions import ArtifactVersionError


@pytest.fixture(scope="module", params=["physchem", "maccs"])
def fitted(request, reference, library):
    eq = ErsiliaQuality(k=5).fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        ignore_size=True,
        scores=ALL_SCORES,
        max_features=4,
        signal_descriptor=request.param,
    )
    return eq


def test_run_columns_and_ranges(fitted, query):
    scores = fitted.run(query).scores
    expected = [
        "typicality",
        "typicality_raw",
        "extremity",
        "extremity_raw",
        "support",
        "support_raw",
        "support_log",
        "consistency",
        "consistency_raw",
        "signal",
        "signal_raw",
    ]
    assert list(scores.columns) == expected
    np.testing.assert_allclose(scores["support_log"], -np.log10(scores["support"]))
    calibrated = scores[["typicality", "extremity", "support", "consistency", "signal"]]
    finite = calibrated.to_numpy()[np.isfinite(calibrated.to_numpy())]
    assert (finite > 0).all() and (finite <= 1).all()


def test_save_load_roundtrip(fitted, query, tmp_path):
    before = fitted.run(query)
    fitted.save(tmp_path / "art")
    loaded = ErsiliaQuality.load(tmp_path / "art")
    after = loaded.run(query)
    pd.testing.assert_frame_equal(before.scores, after.scores)
    assert before.metadata.keys() == after.metadata.keys()


def test_splits_are_not_truncated_by_signal(fitted, reference, tmp_path):
    fitted.save(tmp_path / "art")
    splits = json.loads(
        (tmp_path / "art/reference_mode/shared/splits.json").read_text()
    )
    assert splits["n_train"] + splits["n_val"] + splits["n_test"] == len(reference)
    assert splits["n_train"] == round(0.8 * len(reference))


def test_standalone_component_load(fitted, query, tmp_path):
    fitted.save(tmp_path / "art")
    expected = fitted.run(query).scores
    support = Support.load(tmp_path / "art/reference_mode").run(query)
    np.testing.assert_array_equal(
        support.score.to_numpy(), expected["support"].to_numpy()
    )
    typicality = Typicality.load(tmp_path / "art/reference_mode").run(query)
    np.testing.assert_array_equal(
        typicality.score.to_numpy(), expected["typicality"].to_numpy()
    )


def test_reference_anchors_near_half(fitted):
    for value in (
        fitted.reference_typicality_,
        fitted.reference_extremity_,
        fitted.reference_support_,
        fitted.reference_consistency_,
        fitted.reference_signal_,
    ):
        assert value == pytest.approx(0.5, abs=0.02)


def test_in_library_queries_do_not_match_themselves(fitted, query, library):
    # The last 40 query rows are reference rows 0–39: as queries they must get
    # exactly the nearest-*other*-molecule similarity the library records.
    from eosquality.vectorindex import VectorIndex

    nearest = fitted.run(query).scores["support_raw"].to_numpy()[-40:]
    expected = 1.0 - VectorIndex.load(library).self_knn_distances(1)[:40, 0]
    np.testing.assert_allclose(nearest, expected, atol=1e-6)


def test_support_raw_is_nearest_analogue_similarity(fitted, query):
    support = fitted.support.run(query)
    assert ((support.score_raw >= 0) & (support.score_raw <= 1)).all()
    # Higher similarity never gives lower support.
    order = np.argsort(support.score_raw.to_numpy())
    assert np.all(np.diff(support.score.to_numpy()[order]) >= -1e-12)


def test_old_format_is_rejected(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    meta_path = tmp_path / "art/reference_mode/shared/metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["format_version"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ArtifactVersionError):
        ErsiliaQuality.load(tmp_path / "art")


def test_refit_replaces_all_components(reference, library):
    eq = ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        ignore_size=True,
        scores=["typicality", "support"],
    )
    eq.fit(reference, eos_id="eos0aaa", ignore_size=True, scores=["extremity"])
    assert eq.typicality is None and eq.support is None and eq.extremity is not None


def test_typicality_only_fit_needs_no_index(reference, query):
    eq = ErsiliaQuality().fit(
        reference.drop(columns=["input"]),
        eos_id="eos0aaa",
        ignore_size=True,
        scores=["typicality", "extremity"],
    )
    assert list(eq.run(query.drop(columns=["input"])).scores.columns) == [
        "typicality",
        "typicality_raw",
        "extremity",
        "extremity_raw",
    ]


def test_old_flat_layout_is_rejected(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    (tmp_path / "art/reference_mode").rename(tmp_path / "art/shared")
    with pytest.raises(ArtifactVersionError, match="old flat layout"):
        ErsiliaQuality.load(tmp_path / "art")
