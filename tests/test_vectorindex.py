"""Tests for the VectorIndex module."""

import json

import numpy as np
import pytest

from eosquality.vectorindex.backend import VectorIndex
from tests.conftest import SMILES_20


EOS_ID = "eos4e40"
VERSION = "v1"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_creates_expected_files(tmp_path):
    idx_dir = tmp_path / "idx"
    VectorIndex.build(smiles=SMILES_20, output_dir=idx_dir, max_k=5)
    assert idx_dir.is_dir()
    expected = {"vector_index.h5", "knn_indices.npy", "knn_distances.npy",
                "smiles.csv", "metadata.json"}
    assert {f.name for f in idx_dir.iterdir()} == expected


def test_build_config_records_rdkit_version(tmp_path):
    idx_dir = tmp_path / "idx"
    VectorIndex.build(smiles=SMILES_20, output_dir=idx_dir, max_k=5)
    with open(idx_dir / "metadata.json") as f:
        config = json.load(f)
    assert "rdkit_version" in config
    assert config["rdkit_version"]  # non-empty string


def test_build_config_fields(tmp_path):
    idx_dir = tmp_path / "idx"
    VectorIndex.build(smiles=SMILES_20, output_dir=idx_dir, max_k=7, radius=2, n_bits=512)
    with open(idx_dir / "metadata.json") as f:
        config = json.load(f)
    assert config["max_k"] == 7
    assert config["radius"] == 2
    assert config["n_bits"] == 512


def test_build_raises_if_too_few_molecules(tmp_path):
    idx_dir = tmp_path / "idx"
    with pytest.raises(ValueError, match="max_k"):
        VectorIndex.build(smiles=SMILES_20[:3], output_dir=idx_dir, max_k=5)


def test_build_raises_on_duplicate_smiles(tmp_path):
    idx_dir = tmp_path / "idx"
    smiles_with_dup = SMILES_20[:10] + [SMILES_20[3]]  # one duplicate
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        VectorIndex.build(smiles=smiles_with_dup, output_dir=idx_dir, max_k=5)


def test_build_raises_on_multiple_duplicates(tmp_path):
    idx_dir = tmp_path / "idx"
    smiles_with_dups = SMILES_20[:10] + SMILES_20[:3]  # three duplicates
    with pytest.raises(ValueError, match="3 duplicate"):
        VectorIndex.build(smiles=smiles_with_dups, output_dir=idx_dir, max_k=5)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def test_load_roundtrip(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    assert len(vi._smiles) == len(SMILES_20)
    assert vi._smiles == SMILES_20


def test_load_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "nonexistent")


def test_load_raises_on_file(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="directory"):
        VectorIndex.load(f)


# ---------------------------------------------------------------------------
# validate_smiles
# ---------------------------------------------------------------------------


def test_validate_smiles_passes_exact_match(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    vi.validate_smiles(SMILES_20)  # should not raise


def test_validate_smiles_raises_wrong_order(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    shuffled = SMILES_20[1:] + [SMILES_20[0]]
    with pytest.raises(ValueError, match="mismatch"):
        vi.validate_smiles(shuffled)


def test_validate_smiles_raises_wrong_length(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    with pytest.raises(ValueError, match="count mismatch"):
        vi.validate_smiles(SMILES_20[:10])


# ---------------------------------------------------------------------------
# self_knn_indices / self_knn_distances
# ---------------------------------------------------------------------------


def test_self_knn_indices_shape(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    idx = vi.self_knn_indices(k=5)
    assert idx.shape == (len(SMILES_20), 5)


def test_self_knn_distances_shape(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    dist = vi.self_knn_distances(k=5)
    assert dist.shape == (len(SMILES_20), 5)
    assert (dist >= 0.0).all()
    assert (dist <= 1.0).all()


def test_self_knn_raises_if_k_too_large(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    max_k = vi._config["max_k"]
    with pytest.raises(ValueError, match="max_k"):
        vi.self_knn_indices(k=max_k + 1)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def test_query_shape(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    dists, indices = vi.query(SMILES_20[:5], k=3)
    assert dists.shape == (5, 3)
    assert indices.shape == (5, 3)


def test_query_distances_bounded(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    dists, _ = vi.query(SMILES_20[:5], k=3)
    assert (dists >= 0.0).all()
    assert (dists <= 1.0).all()


def test_query_self_returns_distance_zero(vector_index_dir):
    vi = VectorIndex.load(vector_index_dir)
    # Query each molecule against itself — nearest neighbor should have distance 0
    dists, indices = vi.query([SMILES_20[0]], k=1)
    assert dists[0, 0] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# RDKit version check
# ---------------------------------------------------------------------------


def test_load_raises_on_rdkit_version_mismatch(tmp_path):
    idx_dir = tmp_path / "idx"
    VectorIndex.build(smiles=SMILES_20, output_dir=idx_dir, max_k=5)
    # Tamper with the stored version
    config_path = idx_dir / "metadata.json"
    with open(config_path) as f:
        config = json.load(f)
    config["rdkit_version"] = "0.0.0.fake"
    with open(config_path, "w") as f:
        json.dump(config, f)
    with pytest.raises(RuntimeError, match="RDKit version mismatch"):
        VectorIndex.load(idx_dir)
