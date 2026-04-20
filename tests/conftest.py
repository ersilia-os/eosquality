"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from eosquality.vectorindex.backend import VectorIndex

RNG_SEED = 42

# 20 simple, distinct SMILES for test vector index construction.
# Chosen to be structurally diverse so jaccard kNN produces varied distances.
SMILES_20 = [
    "c1ccccc1",          # benzene
    "c1ccncc1",          # pyridine
    "c1cccnc1",          # pyridine (different N position)
    "c1ccccn1",          # pyridine
    "CC(=O)O",           # acetic acid
    "CCO",               # ethanol
    "CCCO",              # 1-propanol
    "CCCCO",             # 1-butanol
    "CC(C)O",            # isopropanol
    "CC(C)(C)O",         # tert-butanol
    "c1ccc(O)cc1",       # phenol
    "c1ccc(N)cc1",       # aniline
    "c1ccc(Cl)cc1",      # chlorobenzene
    "c1ccc(F)cc1",       # fluorobenzene
    "CC(=O)c1ccccc1",    # acetophenone
    "O=Cc1ccccc1",       # benzaldehyde
    "OC(=O)c1ccccc1",    # benzoic acid
    "Nc1ccccc1",         # aniline (alt form)
    "c1ccc2ccccc2c1",    # naphthalene
    "c1ccoc1",           # furan
]

assert len(SMILES_20) == 20


# ---------------------------------------------------------------------------
# Basic numeric-only fixtures (kept for backward compat with non-FP tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def reference_df(rng):
    """100-row numeric DataFrame drawn from a mixture of two Gaussians."""
    n = 100
    X = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.normal(5, 2, n),
            rng.normal(-3, 0.5, n),
        ]
    )
    return pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])


@pytest.fixture
def query_df(rng):
    """20-row query DataFrame drawn from a similar distribution (in-distribution)."""
    n = 20
    X = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.normal(5, 2, n),
            rng.normal(-3, 0.5, n),
        ]
    )
    return pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])


@pytest.fixture
def ood_query_df(rng):
    """20-row query DataFrame drawn far outside the reference (out-of-distribution)."""
    n = 20
    X = np.column_stack(
        [
            rng.normal(100, 1, n),
            rng.normal(100, 1, n),
            rng.normal(100, 1, n),
        ]
    )
    return pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])


# ---------------------------------------------------------------------------
# Vector-index fixtures (used by the main test suite)
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_df_vi(rng):
    """20-row DataFrame: key (str), input (SMILES), feat_a/b/c (numeric)."""
    n = 20
    X = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.normal(5, 2, n),
            rng.normal(-3, 0.5, n),
        ]
    )
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])
    df.insert(0, "key", [f"key_{i:03d}" for i in range(n)])
    df.insert(1, "input", SMILES_20)
    return df


@pytest.fixture
def query_df_vi(rng):
    """5-row query DataFrame with key, input (first 5 SMILES), numeric features."""
    n = 5
    X = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.normal(5, 2, n),
            rng.normal(-3, 0.5, n),
        ]
    )
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])
    df.insert(0, "key", [f"qkey_{i:03d}" for i in range(n)])
    df.insert(1, "input", SMILES_20[:n])
    return df


@pytest.fixture
def ood_query_df_vi(rng):
    """5-row OOD query DataFrame (features far from reference)."""
    n = 5
    X = np.column_stack(
        [
            rng.normal(100, 1, n),
            rng.normal(100, 1, n),
            rng.normal(100, 1, n),
        ]
    )
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])
    df.insert(0, "key", [f"ood_{i:03d}" for i in range(n)])
    df.insert(1, "input", SMILES_20[:n])
    return df


@pytest.fixture(scope="session")
def vector_index_dir(tmp_path_factory):
    """Session-scoped pre-built VectorIndex for SMILES_20 (built once per session)."""
    idx_dir = tmp_path_factory.mktemp("vi") / "test_index"
    VectorIndex.build(smiles=SMILES_20, output_dir=idx_dir, max_k=15)
    return idx_dir
