import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.cli.build import build_library
from eosquality.library.reference import ReferenceLibrary
from eosquality.scores._match_keys import _scaffold, connectivity_layer


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        library=library,
        max_features=4,
        exclude=["ref_typicality", "ref_extremity"],
    )


def _frame(smiles):
    return pd.DataFrame({"key": [f"q{i}" for i in range(len(smiles))], "input": smiles})


def test_library_member_matches_and_stranger_does_not(fitted, smiles):
    known = smiles[5]
    res = fitted.match.run(_frame([known, "[Fe+2].[Cl-].[Cl-]"]))
    assert res.match.tolist() == [1, 0]
    assert res.match.name == "ref_match" and res.scaffold.name == "ref_scaffold"


def test_match_ignores_stereochemistry_and_salts(fitted, smiles):
    known = smiles[5]
    salt = known + ".Cl"
    res = fitted.match.run(_frame([known, salt]))
    assert res.match.tolist() == [1, 1]


def test_scaffold_is_missing_without_one_and_match_is_missing_if_unparsable(fitted):
    res = fitted.match.run(_frame(["c1ccccc1CC", "CCCCO", "not a smiles"]))
    assert res.scaffold.isna().tolist() == [False, True, True]
    assert res.match.isna().tolist() == [False, False, True]


def test_scaffold_matches_through_the_ring_system(fitted, smiles):
    ringed = next(s for s in smiles[:600] if _scaffold(s))
    res = fitted.match.run(_frame([_scaffold(ringed)]))
    assert res.scaffold.tolist() == [1]


def test_every_reference_molecule_is_in_its_own_library(fitted, smiles):
    res = fitted.match.run(_frame(smiles[:100]))
    assert (res.match == 1).all()


def test_keys_are_library_level(library):
    lib = ReferenceLibrary.load(library)
    molecules, scaffolds = lib.match_keys()
    assert (np.sort(molecules) == molecules).all()
    assert len(set(molecules)) == len(molecules)
    assert 0 < len(scaffolds) <= len(molecules)
    assert lib.library_name == "test_library"
    assert connectivity_layer(lib.smiles[0]) in set(molecules)


def test_missing_keys_give_a_clear_error(tmp_path, smiles):
    build_library(smiles[:10], tmp_path / "lib", "x")
    (tmp_path / "lib" / "connectivity_keys.npz").unlink()
    with pytest.raises(FileNotFoundError, match="match keys"):
        ReferenceLibrary.load(tmp_path / "lib").match_keys()


def test_roundtrip_reads_the_keys_from_the_library(fitted, query, smiles, tmp_path):
    q = query.head(2).copy()
    q["input"] = [smiles[3], "[Fe+2].[Cl-].[Cl-]"]
    before = fitted.run(q)
    fitted.save(tmp_path / "art")
    # The artifact holds no keys: only the counts.
    assert not list((tmp_path / "art/reference_mode/match").glob("*.npz"))
    loaded = ErsiliaQuality.load(tmp_path / "art")
    pd.testing.assert_frame_equal(before.scores, loaded.run(q).scores)
