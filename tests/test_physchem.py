import json

import numpy as np
import pandas as pd
import pytest

from eosquality.library.physchem import (
    DESCRIPTOR_NAMES,
    N_DESCRIPTORS,
    _compute_one,
    check_descriptor_names,
    compute_physchem_raw,
)


def test_descriptors_equal_rdkit_s_own(smiles):
    """The row is exactly what RDKit's own descriptor functions return."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    for smi in smiles[:25]:
        mol = Chem.MolFromSmiles(smi)
        expected = np.array([float(fn(mol)) for _, fn in Descriptors._descList])
        got = _compute_one(smi).astype(np.float64)
        finite = np.isfinite(expected) & (np.abs(expected) < 3e38)
        np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-5, atol=1e-6)
        assert np.isnan(got[~finite]).all()


def test_an_unparsable_smiles_gives_a_row_of_nan():
    row = _compute_one("not a smiles")
    assert row.shape == (N_DESCRIPTORS,) and np.isnan(row).all()


def test_the_matrix_has_one_row_per_molecule_in_order():
    raw = compute_physchem_raw(["CCO", "c1ccccc1", "CCO"])
    assert raw.shape == (3, N_DESCRIPTORS)
    np.testing.assert_array_equal(raw[0], raw[2])
    assert not np.array_equal(raw[0], raw[1])
    assert compute_physchem_raw([]).shape == (0, N_DESCRIPTORS)


def test_a_descriptor_list_that_changed_is_refused():
    check_descriptor_names({"descriptor_names": list(DESCRIPTOR_NAMES)})
    with pytest.raises(RuntimeError, match="descriptor list mismatch"):
        check_descriptor_names({"descriptor_names": DESCRIPTOR_NAMES[:-1]})


# ------------------------------------------------------------ the library cache


def test_the_library_stores_its_descriptors(library):
    from eosquality.library.physchem_cache import PhyschemCache

    cache = PhyschemCache.load(library)
    assert cache is not None and cache.matrix.shape[1] == N_DESCRIPTORS
    assert (np.diff(cache.hashes.astype(np.float64)) > 0).all()  # sorted, unique
    assert json.loads((library / "metadata.json").read_text())["n_physchem"] == len(
        cache.hashes
    )


def test_cached_rows_are_exactly_the_computed_ones(library):
    """Hits, misses and their order: the same matrix with and without the cache."""
    from eosquality.library.physchem_cache import PhyschemCache, describe
    from eosquality.scores._helpers import _standardize_all

    cache = PhyschemCache.load(library)
    known = [
        s
        for s in _standardize_all(open(library / "smiles.csv").read().split()[1:60])
        if s
    ]
    mixed = [known[3], "CCO.CCN", known[0], "CC(C)(C)c1ccccc1O", known[3]]
    expected = compute_physchem_raw(mixed, show_progress=False)
    got = describe(mixed, cache=cache)
    np.testing.assert_array_equal(got, expected)
    rows = cache.rows_of(mixed)
    assert (rows[[0, 2, 4]] >= 0).all() and rows[3] == -1
    np.testing.assert_array_equal(describe(mixed, cache=False), expected)
    assert describe([], cache=cache).shape == (0, N_DESCRIPTORS)


def test_a_cache_from_another_rdkit_or_none_is_ignored(library, tmp_path):
    import shutil

    from eosquality.library.physchem_cache import PhyschemCache

    copy = tmp_path / "lib"
    shutil.copytree(library, copy)
    meta = json.loads((copy / "metadata.json").read_text())
    (copy / "metadata.json").write_text(
        json.dumps({**meta, "rdkit_version": "1999.01"})
    )
    assert PhyschemCache.load(copy) is None
    (copy / "physchem_raw.npy").unlink()
    assert PhyschemCache.load(copy) is None


def test_the_training_scores_read_the_canonical_librarys_cache(
    library, training_dir, monkeypatch, tmp_path
):
    """With the library in reach the training fit uses it, and scores nothing differently."""
    from eosquality import ErsiliaQuality
    from eosquality.library import physchem_cache

    def physchem_scores(env: bool):
        if env:
            monkeypatch.setenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", str(library))
        else:  # no library in reach: neither the env var nor ./data/indices
            monkeypatch.delenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", raising=False)
            monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(physchem_cache, "_loaded", {})
        found = physchem_cache.default_cache() is not None
        eq = ErsiliaQuality().fit(training_sets=training_dir, eos_id="eos0aaa")
        queries = pd.DataFrame({"input": ["CCO", "CC(C)(C)c1ccccc1O"]})
        return found, eq.run(queries).scores

    with_cache, scores = physchem_scores(True)
    without_cache, expected = physchem_scores(False)
    assert with_cache and not without_cache
    pd.testing.assert_frame_equal(scores, expected)


def test_a_huge_molecule_skips_the_polynomial_descriptors_quickly():
    import time

    from eosquality.library.physchem import DESCRIPTOR_NAMES, _compute_one

    started = time.perf_counter()
    row = _compute_one("C" * 400)
    assert time.perf_counter() - started < 5
    for name in ("Ipc", "AvgIpc"):
        assert np.isnan(row[DESCRIPTOR_NAMES.index(name)])
    assert np.isfinite(row[DESCRIPTOR_NAMES.index("MolWt")])
