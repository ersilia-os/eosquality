import pytest

from eosquality.vectorindex import VectorIndex


def test_build_refuses_to_resume_with_different_smiles(tmp_path, smiles):
    VectorIndex.build(smiles[:100], tmp_path / "idx", max_k=5)
    with pytest.raises(ValueError, match="different inputs"):
        VectorIndex.build(smiles[1:101], tmp_path / "idx", max_k=5)


def test_build_resumes_with_identical_inputs(tmp_path, smiles):
    VectorIndex.build(smiles[:100], tmp_path / "idx", max_k=5)
    vi = VectorIndex.build(smiles[:100], tmp_path / "idx", max_k=5)
    assert vi.n_reference == 100


def test_self_knn_excludes_self(tmp_path, smiles):
    vi = VectorIndex.build(smiles[:100], tmp_path / "idx", max_k=5)
    idx = vi.self_knn_indices(5)
    assert (idx != __import__("numpy").arange(len(idx))[:, None]).all()
