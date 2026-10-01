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
    splits = json.loads((tmp_path / "art/shared/splits.json").read_text())
    assert splits["n_train"] + splits["n_val"] + splits["n_test"] == len(reference)
    assert splits["n_train"] == round(0.8 * len(reference))


def test_standalone_component_load(fitted, query, tmp_path):
    fitted.save(tmp_path / "art")
    expected = fitted.run(query).scores
    support = Support.load(tmp_path / "art").run(query)
    np.testing.assert_array_equal(
        support.score.to_numpy(), expected["support"].to_numpy()
    )
    typicality = Typicality.load(tmp_path / "art").run(query)
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


def test_in_library_queries_do_not_match_themselves(fitted, query, reference):
    # The last 40 query rows are reference molecules: their support must be
    # computed against 5 *other* molecules, like the reference's own CDF.
    support = fitted.run(query).scores["support_raw"].to_numpy()[-40:]
    assert (support > 0).all()


def test_old_format_is_rejected(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    meta_path = tmp_path / "art/shared/metadata.json"
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


def test_support_is_calibrated_within_each_size_bin(fitted, reference, library):
    from eosquality.scores._binning import assign_bins
    from eosquality.vectorindex import VectorIndex

    support = fitted.support
    sizes = VectorIndex.load(library).fingerprint_sizes()
    distances = np.concatenate(support.sorted_self_distances_per_bin_)
    assert distances.size == len(reference)
    bins = assign_bins(sizes, support.size_bin_edges_)
    for b, sorted_arr in enumerate(support.sorted_self_distances_per_bin_):
        assert sorted_arr.size == (bins == b).sum()
    assert support.n_bins_ > 1
