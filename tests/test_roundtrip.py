import json

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality, Support, Typicality
from eosquality.exceptions import ArtifactVersionError

REFERENCE = ["typicality", "extremity", "support", "consistency", "signal"]


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, max_features=4
    )


def test_run_columns_and_ranges(fitted, query):
    scores = fitted.run(query).scores
    expected = [
        "ref_typicality",
        "ref_typicality_raw",
        "ref_extremity",
        "ref_extremity_raw",
        "ref_support",
        "ref_support_raw",
        "ref_support_log",
        "ref_consistency",
        "ref_consistency_raw",
        "ref_signal",
        "ref_signal_raw",
    ]
    assert list(scores.columns) == expected
    np.testing.assert_allclose(
        scores["ref_support_log"], -np.log10(scores["ref_support"])
    )
    calibrated = scores[[f"ref_{name}" for name in REFERENCE]]
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
        support.score.to_numpy(), expected["ref_support"].to_numpy()
    )
    typicality = Typicality.load(tmp_path / "art/reference_mode").run(query)
    np.testing.assert_array_equal(
        typicality.score.to_numpy(), expected["ref_typicality"].to_numpy()
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

    nearest = fitted.run(query).scores["ref_support_raw"].to_numpy()[-40:]
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
    only = ["ref_typicality", "ref_support"]
    eq = ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        exclude=[f"ref_{n}" for n in REFERENCE if f"ref_{n}" not in only],
    )
    eq.fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        exclude=[f"ref_{n}" for n in REFERENCE if n != "extremity"],
    )
    assert eq.typicality is None and eq.support is None and eq.extremity is not None


def test_output_scores_run_without_smiles(reference, library, query):
    eq = ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        exclude=["ref_support", "ref_consistency", "ref_signal"],
    )
    assert list(eq.run(query.drop(columns=["input"])).scores.columns) == [
        "ref_typicality",
        "ref_typicality_raw",
        "ref_extremity",
        "ref_extremity_raw",
    ]


def test_reference_must_match_the_library(reference, library):
    with pytest.raises(ValueError, match="SMILES"):
        ErsiliaQuality().fit(
            reference.iloc[::-1], eos_id="eos0aaa", vector_index=library
        )


def test_excluding_everything_is_an_error(reference, library):
    with pytest.raises(ValueError, match="nothing to fit"):
        ErsiliaQuality().fit(
            reference,
            eos_id="eos0aaa",
            vector_index=library,
            exclude=[f"ref_{n}" for n in REFERENCE],
        )
    with pytest.raises(ValueError, match="Unknown score"):
        ErsiliaQuality().fit(
            reference, eos_id="eos0aaa", vector_index=library, exclude=["support"]
        )


def test_old_flat_layout_is_rejected(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    (tmp_path / "art/reference_mode").rename(tmp_path / "art/shared")
    with pytest.raises(ArtifactVersionError, match="old flat layout"):
        ErsiliaQuality.load(tmp_path / "art")


def test_metadata_keys_are_stable(fitted, query):
    """The public metadata contract (docs/api.md)."""
    metadata = fitted.run(query).metadata
    assert metadata["n_reference"] == 600
    for score in REFERENCE:
        assert metadata[f"ref_{score}_anchor"] == pytest.approx(0.5, abs=0.02)
    assert {"ref_support_k", "ref_consistency_k"} <= set(metadata)
    assert metadata["ref_signal_descriptor"] == "physchem"
    # No key repeats its own score name, and none is left unprefixed.
    for key in metadata:
        if key == "n_reference":
            continue
        assert key.startswith(("ref_", "trn_")), key
        score = "_".join(key.split("_")[:2])
        assert not key[len(score) + 1 :].startswith(score.split("_")[1]), key
