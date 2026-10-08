import json

import numpy as np
import pytest

from eosquality.vectorindex import VectorIndex


@pytest.fixture(scope="module")
def built(tmp_path_factory, smiles):
    folder = tmp_path_factory.mktemp("index") / "idx"
    return VectorIndex.build(smiles[:100], folder, max_k=5), folder


def test_build_writes_the_index_folder(built):
    _, folder = built
    assert sorted(p.name for p in folder.iterdir()) == [
        "knn_distances.npy",
        "metadata.json",
        "smiles.csv",
        "vector_index.h5",
    ]


def test_load_gives_the_same_index(built, smiles):
    vi, folder = built
    loaded = VectorIndex.load(folder)
    assert loaded.smiles == vi.smiles == smiles[:100]
    np.testing.assert_array_equal(
        loaded.self_knn_distances(3), vi.self_knn_distances(5)[:, :3]
    )
    assert loaded.index_dir == folder


def test_self_knn_distances_leave_the_molecule_itself_out(built, smiles):
    """The nearest *other* molecule is never at distance 0 for unique molecules."""
    distances = built[0].self_knn_distances(5)
    assert distances.shape == (100, 5) and (distances > 0).all()
    assert (np.diff(distances, axis=1) >= 0).all()


def test_query_finds_a_molecule_as_its_own_nearest(built, smiles):
    distances, indices = built[0].query(smiles[:10], k=3)
    assert distances.shape == indices.shape == (10, 3)
    np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-6)
    assert (indices[:, 0] == np.arange(10)).all()
    assert (np.diff(distances, axis=1) >= 0).all()


def test_k_beyond_the_precomputed_depth_is_refused(built):
    with pytest.raises(ValueError, match="exceeds the pre-computed max_k=5"):
        built[0].self_knn_distances(6)


def test_build_needs_unique_smiles_and_enough_molecules(tmp_path, smiles):
    with pytest.raises(ValueError, match="Duplicate SMILES"):
        VectorIndex.build([*smiles[:20], smiles[0]], tmp_path / "a", max_k=5)
    with pytest.raises(ValueError, match="at least max_k"):
        VectorIndex.build(smiles[:5], tmp_path / "b", max_k=5)


def test_another_rdkit_version_is_refused(built, tmp_path):
    import shutil

    _, folder = built
    other = tmp_path / "idx"
    shutil.copytree(folder, other)
    meta = json.loads((other / "metadata.json").read_text())
    meta["rdkit_version"] = "1999.01.1"
    (other / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="RDKit version mismatch"):
        VectorIndex.load(other)


def test_loading_a_missing_folder_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No vector index folder"):
        VectorIndex.load(tmp_path / "nope")
