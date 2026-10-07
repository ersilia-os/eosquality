import json

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality, Typicality
from eosquality.exceptions import ArtifactVersionError

REFERENCE = ["typicality", "extremity", "match"]
PCT_SCORES = ("typicality", "extremity")


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", library=library, max_features=4
    )


def test_run_columns_and_ranges(fitted, query):
    scores = fitted.run(query).scores
    assert list(scores.columns) == [
        "ref_typicality_pct",
        "ref_typicality_raw",
        "ref_extremity_pct",
        "ref_extremity_raw",
        "ref_match",
        "ref_scaffold",
    ]
    calibrated = scores[[f"ref_{n}_pct" for n in PCT_SCORES]]
    finite = calibrated.to_numpy()[np.isfinite(calibrated.to_numpy())]
    assert (finite > 0).all() and (finite <= 1).all()
    raw = scores["ref_extremity_raw"].dropna()
    assert ((raw >= 0) & (raw <= 1)).all()
    assert set(scores["ref_match"].dropna()) <= {0, 1}


def test_save_load_roundtrip(fitted, query, tmp_path):
    before = fitted.run(query)
    fitted.save(tmp_path / "art")
    loaded = ErsiliaQuality.load(tmp_path / "art")
    after = loaded.run(query)
    pd.testing.assert_frame_equal(before.scores, after.scores)
    assert before.metadata.keys() == after.metadata.keys()


def test_artifacts_hold_no_splits_or_scaled_reference(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    shared = tmp_path / "art/reference_mode/shared"
    assert not (shared / "splits.json").exists()
    assert not (shared / "reference_repr.npy").exists()
    assert not (tmp_path / "art/reference_mode/knn").exists()


def test_standalone_component_load(fitted, query, tmp_path):
    fitted.save(tmp_path / "art")
    expected = fitted.run(query).scores
    typicality = Typicality.load(tmp_path / "art/reference_mode").run(query)
    np.testing.assert_array_equal(
        typicality.score.to_numpy(), expected["ref_typicality_pct"].to_numpy()
    )


def test_reference_anchors_near_half(fitted):
    for value in (fitted.reference_typicality_, fitted.reference_extremity_):
        assert value == pytest.approx(0.5, abs=0.02)


def test_old_format_is_rejected(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    meta_path = tmp_path / "art/reference_mode/shared/metadata.json"
    meta = json.loads(meta_path.read_text())
    del meta["format_version"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ArtifactVersionError):
        ErsiliaQuality.load(tmp_path / "art")


def test_refit_replaces_all_components(reference, library):
    exclude = lambda keep: [f"ref_{n}" for n in REFERENCE if n not in keep]  # noqa: E731
    eq = ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        library=library,
        exclude=exclude(["typicality", "match"]),
    )
    eq.fit(reference, eos_id="eos0aaa", library=library, exclude=exclude(["extremity"]))
    assert eq.typicality is None and eq.match is None and eq.extremity is not None


def test_output_scores_run_without_smiles(reference, library, query):
    eq = ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", library=library, exclude=["ref_match"]
    )
    assert list(eq.run(query.drop(columns=["input"])).scores.columns) == [
        "ref_typicality_pct",
        "ref_typicality_raw",
        "ref_extremity_pct",
        "ref_extremity_raw",
    ]


def test_reference_must_match_the_library(reference, library):
    with pytest.raises(ValueError, match="SMILES"):
        ErsiliaQuality().fit(reference.iloc[::-1], eos_id="eos0aaa", library=library)


def test_excluding_everything_is_an_error(reference, library):
    with pytest.raises(ValueError, match="nothing to fit"):
        ErsiliaQuality().fit(
            reference,
            eos_id="eos0aaa",
            library=library,
            exclude=[f"ref_{n}" for n in REFERENCE],
        )
    with pytest.raises(ValueError, match="Unknown score"):
        ErsiliaQuality().fit(
            reference, eos_id="eos0aaa", library=library, exclude=["support"]
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
    for score in PCT_SCORES:
        assert metadata[f"ref_{score}_anchor"] == pytest.approx(0.5, abs=0.02)
    assert metadata["ref_match_n_molecules"] > 0
    assert metadata["ref_match_n_scaffolds"] > 0
    # No key repeats its own score name, and none is left unprefixed.
    for key in metadata:
        if key == "n_reference":
            continue
        assert key.startswith(("ref_", "trn_")), key
        score = "_".join(key.split("_")[:2])
        assert not key[len(score) + 1 :].startswith(score.split("_")[1]), key
