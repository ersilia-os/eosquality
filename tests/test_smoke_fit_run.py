"""Smoke tests covering the end-to-end fit/run/save/load workflow."""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality, RunResult
from eosquality.exceptions import NotFittedError, SchemaError
from eosquality.utils.identifiers import extract_from_path, validate_eos_id, validate_version


EXPECTED_SCORE_COLUMNS = {
    "quality_score",
    "support_score",
    "typicality_score",
    "consistency_score",
    "distance_k_mean",
    "distance_k_max",
    "nearest_reference_ids",
}

EOS_ID = "eos4e40"
VERSION = "v1"


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_completes(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10)
    result = eq.fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                    vector_index=vector_index_dir, ignore_size=True)
    assert result is eq, "fit() should return self"
    assert eq.is_fitted_


def test_fit_exposes_schema(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    expected_cols = ["feat_a", "feat_b", "feat_c"]
    assert eq.schema_.column_names == expected_cols


def test_fit_exposes_reference_quality(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    assert 0.0 < eq.reference_quality_ <= 1.0


def test_fit_exposes_reference_report(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    report = eq.reference_report_
    assert 0.0 <= report.cohesion_score <= 1.0
    assert report.fragmentation_score >= 0.0


def test_fit_strips_key_input_silently(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    assert "key" not in eq.schema_.column_names
    assert "input" not in eq.schema_.column_names
    assert eq.schema_.column_names == ["feat_a", "feat_b", "feat_c"]


def test_fit_raises_on_unexpected_non_numeric_columns(reference_df_vi, vector_index_dir):
    df = reference_df_vi.copy()
    df["label"] = "hello"
    with pytest.raises(SchemaError, match="Unexpected non-numeric"):
        ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


def test_fit_raises_when_all_columns_non_numeric(vector_index_dir):
    from tests.conftest import SMILES_20
    df = pd.DataFrame({"key": [f"k{i}" for i in range(20)], "input": SMILES_20})
    with pytest.raises(SchemaError, match="No numeric columns"):
        ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


def test_fit_raises_on_empty_df(vector_index_dir):
    with pytest.raises(SchemaError, match="empty"):
        ErsiliaQuality(k=10).fit(pd.DataFrame(), eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


def test_fit_raises_on_small_dataset(reference_df_vi, vector_index_dir):
    with pytest.raises(ValueError, match="10,000"):
        ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=False)


def test_fit_falls_back_to_reference_library_when_omitted(reference_df_vi):
    """With no vector_index passed, fit() uses the canonical library CSV for
    its SMILES-alignment check. The 20 hand-picked SMILES in the test fixture
    don't match the canonical library, so we expect the CSV-based mismatch
    error — confirming the canonical resolver path was exercised.

    Runs network-free: the repo's ``data/libraries/<LIBRARY_ID>.csv`` satisfies
    the CWD fallback in reference_library_csv_path().
    """
    with pytest.raises(ValueError, match="SMILES mismatch against canonical library"):
        ErsiliaQuality(k=10).fit(
            reference_df_vi, eos_id=EOS_ID, version=VERSION, ignore_size=True
        )


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_fit_metadata_fields(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    m = eq.metadata_
    assert m.eos_id == EOS_ID
    assert m.version == VERSION
    assert m.n_samples == len(reference_df_vi)
    # n_features counts only the numeric columns
    assert m.n_features == 3
    assert m.columns == ["feat_a", "feat_b", "feat_c"]
    assert set(m.column_stats.keys()) == {"feat_a", "feat_b", "feat_c"}
    for col, stats in m.column_stats.items():
        assert set(stats.keys()) == {"mean", "std", "min", "max", "median"}
    assert set(m.missing_counts.keys()) == {"feat_a", "feat_b", "feat_c"}
    assert m.fit_timestamp  # non-empty ISO string
    assert m.eosquality_version  # non-empty string


# ---------------------------------------------------------------------------
# column characteristics
# ---------------------------------------------------------------------------


def _fit_with_columns(col_dict, vector_index_dir):
    """Helper: build a 20-row DataFrame with SMILES + given numeric columns and fit."""
    from tests.conftest import SMILES_20
    df = pd.DataFrame(col_dict)
    df.insert(0, "key", [f"k{i}" for i in range(20)])
    df.insert(1, "input", SMILES_20)
    eq = ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    return eq.metadata_.column_characteristics


def test_fit_metadata_has_column_characteristics(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    chars = eq.metadata_.column_characteristics
    assert set(chars.keys()) == {"feat_a", "feat_b", "feat_c"}
    for c in chars.values():
        assert hasattr(c, "kind")
        assert hasattr(c, "sparsity")
        assert hasattr(c, "missing_fraction")
        assert c.kind in {"binary", "count", "continuous"}
        assert 0.0 <= c.sparsity <= 1.0
        assert 0.0 <= c.missing_fraction <= 1.0


def test_column_kind_binary(vector_index_dir):
    import numpy as np
    rng = np.random.default_rng(0)
    chars = _fit_with_columns({"x": rng.integers(0, 2, 20).astype(float)}, vector_index_dir)
    assert chars["x"].kind == "binary"


def test_column_kind_proportion_is_continuous(vector_index_dir):
    import numpy as np
    rng = np.random.default_rng(1)
    # values in (0, 1) that are not binary → continuous (proportions are not a separate kind)
    vals = rng.uniform(0.05, 0.95, 20)
    chars = _fit_with_columns({"x": vals}, vector_index_dir)
    assert chars["x"].kind == "continuous"


def test_column_kind_count(vector_index_dir):
    import numpy as np
    rng = np.random.default_rng(2)
    # non-negative integers, some > 1 → count
    vals = rng.integers(0, 50, 20).astype(float)
    # ensure at least one value > 1 to avoid binary classification
    vals[0] = 5.0
    chars = _fit_with_columns({"x": vals}, vector_index_dir)
    assert chars["x"].kind == "count"


def test_column_kind_continuous(vector_index_dir):
    import numpy as np
    rng = np.random.default_rng(3)
    # values with negatives → continuous
    vals = rng.normal(-5, 3, 20)
    chars = _fit_with_columns({"x": vals}, vector_index_dir)
    assert chars["x"].kind == "continuous"


def test_column_sparsity(vector_index_dir):
    import numpy as np
    # exactly 10 zeros out of 20 → sparsity = 0.5
    vals = np.array([0.0] * 10 + [1.5] * 10)
    chars = _fit_with_columns({"x": vals}, vector_index_dir)
    assert abs(chars["x"].sparsity - 0.5) < 1e-9


def test_column_missing_fraction(vector_index_dir):
    import numpy as np
    # 5 NaN out of 20 → missing_fraction = 0.25
    vals = np.array([1.0] * 15 + [float("nan")] * 5)
    chars = _fit_with_columns({"x": vals}, vector_index_dir)
    assert abs(chars["x"].missing_fraction - 0.25) < 1e-9


def test_metadata_roundtrip_includes_characteristics(reference_df_vi, vector_index_dir, tmp_path):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    folder = tmp_path / "ref"
    eq.save(folder)
    eq2 = ErsiliaQuality.load(folder)
    chars_before = eq.metadata_.column_characteristics
    chars_after = eq2.metadata_.column_characteristics
    assert set(chars_before.keys()) == set(chars_after.keys())
    for col in chars_before:
        assert chars_before[col].kind == chars_after[col].kind
        assert abs(chars_before[col].sparsity - chars_after[col].sparsity) < 1e-9
        assert abs(chars_before[col].missing_fraction - chars_after[col].missing_fraction) < 1e-9


# ---------------------------------------------------------------------------
# duplicate detection
# ---------------------------------------------------------------------------


def test_fit_raises_on_wrong_smiles_alignment(reference_df_vi, vector_index_dir):
    """Reference CSV whose SMILES diverge from the library must be rejected."""
    df = reference_df_vi.copy()
    df.iloc[2] = df.iloc[1]
    with pytest.raises(ValueError, match="SMILES mismatch"):
        ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


def test_fit_raises_on_duplicate_keys(reference_df_vi, vector_index_dir):
    df = reference_df_vi.copy()
    # make two rows share the same key but different SMILES
    df = df.copy()
    df.at[df.index[2], "key"] = df.at[df.index[1], "key"]
    with pytest.raises(ValueError, match="Duplicate key"):
        ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


# ---------------------------------------------------------------------------
# SMILES pre-checks
# ---------------------------------------------------------------------------


def test_fit_raises_on_nan_smiles(reference_df_vi, vector_index_dir):
    import numpy as np
    df = reference_df_vi.copy()
    df.at[df.index[0], "input"] = np.nan
    with pytest.raises(SchemaError, match="NaN"):
        ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


def test_fit_raises_on_empty_smiles(reference_df_vi, vector_index_dir):
    df = reference_df_vi.copy()
    df.at[df.index[0], "input"] = ""
    with pytest.raises(SchemaError, match="empty string"):
        ErsiliaQuality(k=10).fit(df, eos_id=EOS_ID, version=VERSION,
                                 vector_index=vector_index_dir, ignore_size=True)


def test_fit_invalid_eos_id(reference_df_vi, vector_index_dir):
    for bad_id in ("eos", "abc1234", "", "eos12345", "EOS4e40"):
        with pytest.raises(ValueError, match="EOS identifier"):
            ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=bad_id, version=VERSION,
                                     vector_index=vector_index_dir)


def test_fit_invalid_version(reference_df_vi, vector_index_dir):
    for bad_ver in ("1", "version1", "V1", "v", ""):
        with pytest.raises(ValueError, match="version"):
            ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=bad_ver,
                                     vector_index=vector_index_dir)


# ---------------------------------------------------------------------------
# identifier utilities
# ---------------------------------------------------------------------------


def test_validate_eos_id_valid():
    for good in ("eos4e40", "eos7m30", "eos3804", "eos42ez"):
        validate_eos_id(good)  # should not raise


def test_validate_eos_id_invalid():
    for bad in ("eos", "abc1234", "EOS4e40", "eos12345", "eos4e4"):
        with pytest.raises(ValueError):
            validate_eos_id(bad)


def test_validate_version_valid():
    for good in ("v1", "v2", "v10", "v100"):
        validate_version(good)  # should not raise


def test_validate_version_invalid():
    for bad in ("1", "V1", "version1", "v", "v1a"):
        with pytest.raises(ValueError):
            validate_version(bad)


def test_extract_from_path_canonical():
    eos_id, version = extract_from_path("eos4e40_v1.csv")
    assert eos_id == "eos4e40"
    assert version == "v1"


def test_extract_from_path_with_prefix():
    eos_id, version = extract_from_path("260313_project_eos4e40_v2.csv")
    assert eos_id == "eos4e40"
    assert version == "v2"


def test_extract_from_path_missing_version():
    with pytest.raises(ValueError, match="version"):
        extract_from_path("eos4e40.csv")


def test_extract_from_path_no_eos_id():
    with pytest.raises(ValueError, match="EOS identifier"):
        extract_from_path("reference.csv")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_returns_run_result(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    result = eq.run(query_df_vi)
    assert isinstance(result, RunResult)


def test_run_scores_shape(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    result = eq.run(query_df_vi)
    assert result.scores.shape[0] == len(query_df_vi)


def test_run_score_columns(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    result = eq.run(query_df_vi)
    assert EXPECTED_SCORE_COLUMNS.issubset(set(result.scores.columns))


def test_run_scores_bounded(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    result = eq.run(query_df_vi)
    for col in ("quality_score", "support_score", "typicality_score",
                "consistency_score"):
        assert (result.scores[col] >= 0.0).all(), f"{col} has values < 0"
        assert (result.scores[col] <= 1.0).all(), f"{col} has values > 1"


def test_anchor_equals_normalized_median():
    """A value at the reference median normalizes to the stored anchor."""
    from eosquality.preprocess.pipeline import _fit_type_aware

    rng = np.random.default_rng(7)
    values = rng.normal(50.0, 10.0, 200)
    params, _ = _fit_type_aware(values)
    p50_raw = float(np.median(values))
    p50_norm = np.clip((p50_raw - params["p1"]) / (params["p99"] - params["p1"]), 0.0, 1.0)
    assert abs(p50_norm - params["anchor"]) < 1e-6


def test_run_preserves_index(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    result = eq.run(query_df_vi)
    assert list(result.scores.index) == list(query_df_vi.index)


def test_run_ood_scores_lower_than_in_distribution(
    reference_df_vi, query_df_vi, ood_query_df_vi, vector_index_dir
):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    in_dist = eq.run(query_df_vi).scores["quality_score"].mean()
    out_dist = eq.run(ood_query_df_vi).scores["quality_score"].mean()
    assert out_dist < in_dist, "OOD queries should have lower quality scores"


def test_run_metadata(reference_df_vi, query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    result = eq.run(query_df_vi)
    assert "reference_quality" in result.metadata
    assert result.metadata["k"] == 10


def test_run_raises_when_not_fitted(query_df_vi):
    with pytest.raises(NotFittedError):
        ErsiliaQuality(k=10).run(query_df_vi)


def test_run_raises_on_missing_columns(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    with pytest.raises(SchemaError, match="missing"):
        eq.run(pd.DataFrame({"feat_a": [1.0], "input": ["c1ccccc1"]}))


def test_run_raises_without_input_column(reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    with pytest.raises(SchemaError, match="input"):
        eq.run(pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "feat_c": [3.0]}))


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(reference_df_vi, query_df_vi, vector_index_dir, tmp_path):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    scores_before = eq.run(query_df_vi).scores

    artifact_folder = tmp_path / "reference"
    eq.save(artifact_folder)
    assert artifact_folder.is_dir()
    expected_files = {
        "config.json", "schema.json", "reference_report.json", "metadata.json",
        "reference_ids.joblib", "scalers.joblib",
        "reference_repr.npy", "reference_knn_distances.npy", "reference_knn_indices.npy",
    }
    assert expected_files == {f.name for f in artifact_folder.iterdir()}

    eq2 = ErsiliaQuality.load(artifact_folder)
    assert eq2.is_fitted_

    scores_after = eq2.run(query_df_vi).scores
    pd.testing.assert_frame_equal(
        scores_before.drop(columns=["nearest_reference_ids"]),
        scores_after.drop(columns=["nearest_reference_ids"]),
    )


def test_metadata_in_artifact_folder(reference_df_vi, vector_index_dir, tmp_path):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    folder = tmp_path / "ref"
    eq.save(folder)
    with open(folder / "metadata.json") as f:
        data = json.load(f)
    assert data["eos_id"] == EOS_ID
    assert data["version"] == VERSION
    assert data["n_samples"] == len(reference_df_vi)
    assert data["n_features"] == 3
    assert "column_stats" in data
    assert "missing_counts" in data
    assert "fit_timestamp" in data
    assert "eosquality_version" in data


def test_metadata_roundtrip(reference_df_vi, vector_index_dir, tmp_path):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    folder = tmp_path / "ref"
    eq.save(folder)
    eq2 = ErsiliaQuality.load(folder)
    assert eq2.metadata_.eos_id == EOS_ID
    assert eq2.metadata_.version == VERSION
    assert eq2.metadata_.n_samples == len(reference_df_vi)


def test_save_creates_parent_dirs(reference_df_vi, vector_index_dir, tmp_path):
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    folder = tmp_path / "nested" / "dir" / "reference"
    eq.save(folder)
    assert folder.is_dir()


def test_save_raises_when_not_fitted(tmp_path):
    with pytest.raises(NotFittedError):
        ErsiliaQuality(k=10).save(tmp_path / "reference")


def test_load_raises_on_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        ErsiliaQuality.load(tmp_path / "nonexistent")


def test_load_raises_on_file_instead_of_folder(reference_df_vi, vector_index_dir, tmp_path):
    import joblib
    p = tmp_path / "not_a_folder.joblib"
    joblib.dump({"dummy": True}, p)
    with pytest.raises(ValueError, match="directory"):
        ErsiliaQuality.load(p)


# ---------------------------------------------------------------------------
# type_aware preprocessing strategy
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_reference_df_vi():
    """20-row DataFrame with binary, count, and continuous columns + SMILES."""
    from tests.conftest import SMILES_20
    rng = np.random.default_rng(99)
    n = 20
    df = pd.DataFrame({
        "key": [f"k{i}" for i in range(n)],
        "input": SMILES_20,
        "feat_binary": rng.integers(0, 2, n).astype(float),
        "feat_count": rng.integers(0, 100, n).astype(float),
        "feat_continuous": rng.normal(0, 5, n),
    })
    return df


@pytest.fixture
def mixed_query_df_vi():
    """5-row query with same column layout."""
    from tests.conftest import SMILES_20
    rng = np.random.default_rng(100)
    n = 5
    return pd.DataFrame({
        "key": [f"qk{i}" for i in range(n)],
        "input": SMILES_20[:n],
        "feat_binary": rng.integers(0, 2, n).astype(float),
        "feat_count": rng.integers(0, 100, n).astype(float),
        "feat_continuous": rng.normal(0, 5, n),
    })


def test_type_aware_fit_completes(mixed_reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(
        mixed_reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    assert eq.is_fitted_


def test_type_aware_scores_bounded(mixed_reference_df_vi, mixed_query_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(
        mixed_reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    result = eq.run(mixed_query_df_vi)
    for col in ("quality_score", "support_score", "typicality_score",
                "consistency_score"):
        assert (result.scores[col] >= 0.0).all(), f"{col} has values < 0"
        assert (result.scores[col] <= 1.0).all(), f"{col} has values > 1"


def test_type_aware_column_kinds_detected(mixed_reference_df_vi, vector_index_dir):
    eq = ErsiliaQuality(k=10).fit(
        mixed_reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    chars = eq.metadata_.column_characteristics
    assert chars["feat_binary"].kind == "binary"
    assert chars["feat_count"].kind == "count"
    assert chars["feat_continuous"].kind == "continuous"


def test_type_aware_repr_in_unit_interval(mixed_reference_df_vi, vector_index_dir):
    """After fit, the stored reference_repr should be in [0, 1] for type_aware."""
    eq = ErsiliaQuality(k=10).fit(
        mixed_reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    arr = eq._fit_state.reference_repr
    assert arr.min() >= 0.0 - 1e-9
    assert arr.max() <= 1.0 + 1e-9


def test_type_aware_save_load_roundtrip(
    mixed_reference_df_vi, mixed_query_df_vi, vector_index_dir, tmp_path
):
    eq = ErsiliaQuality(k=10).fit(
        mixed_reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    scores_before = eq.run(mixed_query_df_vi).scores

    folder = tmp_path / "ta_ref"
    eq.save(folder)
    eq2 = ErsiliaQuality.load(folder)

    scores_after = eq2.run(mixed_query_df_vi).scores
    pd.testing.assert_frame_equal(
        scores_before.drop(columns=["nearest_reference_ids"]),
        scores_after.drop(columns=["nearest_reference_ids"]),
    )


def test_type_aware_normalizer_params_persisted(
    mixed_reference_df_vi, vector_index_dir, tmp_path
):
    """scalers.joblib must contain use_log1p/p1/p99/anchor for every feature column."""
    import joblib
    eq = ErsiliaQuality(k=10).fit(
        mixed_reference_df_vi, eos_id=EOS_ID, version=VERSION,
        vector_index=vector_index_dir, ignore_size=True,
    )
    folder = tmp_path / "ta_params"
    eq.save(folder)

    scalers = joblib.load(folder / "scalers.joblib")
    for col in ["feat_binary", "feat_count", "feat_continuous"]:
        assert col in scalers
        p = scalers[col]
        assert "use_log1p" in p
        assert "p1" in p
        assert "p99" in p
        assert "anchor" in p
        assert isinstance(p["use_log1p"], bool)
        assert isinstance(p["p1"], float)
        assert isinstance(p["p99"], float)
        assert 0.0 <= p["anchor"] <= 1.0


# ---------------------------------------------------------------------------
# distribution-based normalization: anchor and log1p trigger
# ---------------------------------------------------------------------------


def test_scaler_params_have_anchor(reference_df_vi, vector_index_dir):
    """Every scaler entry includes an 'anchor' key in [0, 1]."""
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    state = eq._fit_state.preprocess_state
    for col, params in state["scalers"].items():
        assert "anchor" in params, f"Missing anchor for column {col}"
        assert 0.0 <= params["anchor"] <= 1.0


def test_scaler_params_have_use_log1p(reference_df_vi, vector_index_dir):
    """Every scaler entry includes a 'use_log1p' boolean key."""
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    state = eq._fit_state.preprocess_state
    for col, params in state["scalers"].items():
        assert "use_log1p" in params, f"Missing use_log1p for column {col}"
        assert isinstance(params["use_log1p"], bool)


def test_log1p_trigger_sparse_large_range():
    """Distribution-based trigger fires for sparse high-range data."""
    from eosquality.preprocess.pipeline import _fit_type_aware
    rng = np.random.default_rng(42)
    values = np.zeros(100)
    values[10:] = rng.uniform(1.0, 1000.0, 90)  # 10% zeros, range ~1000x
    params, _ = _fit_type_aware(values)
    assert params["use_log1p"] is True


def test_log1p_trigger_not_sparse():
    """Distribution-based trigger does not fire for non-sparse data."""
    from eosquality.preprocess.pipeline import _fit_type_aware
    rng = np.random.default_rng(42)
    values = rng.uniform(1.0, 1000.0, 100)  # no zeros
    params, _ = _fit_type_aware(values)
    assert params["use_log1p"] is False


def test_log1p_trigger_small_range():
    """Distribution-based trigger does not fire when range ratio <= 10."""
    from eosquality.preprocess.pipeline import _fit_type_aware
    # 50% zeros but non-zero values only span 1–5 (ratio = 5)
    values = np.array([0.0] * 50 + [1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    params, _ = _fit_type_aware(values)
    assert params["use_log1p"] is False


def test_anchor_bell_shaped():
    """Bell-shaped distribution → anchor near 0.5."""
    from eosquality.preprocess.pipeline import _fit_type_aware
    rng = np.random.default_rng(42)
    values = rng.normal(500.0, 50.0, 1000)  # symmetric bell
    params, _ = _fit_type_aware(values)
    assert 0.35 <= params["anchor"] <= 0.65, (
        f"Expected anchor ~0.5 for bell-shaped data, got {params['anchor']:.3f}"
    )


def test_anchor_right_skewed():
    """Right-skewed / sparse distribution → anchor significantly below 0.5."""
    from eosquality.preprocess.pipeline import _fit_type_aware
    rng = np.random.default_rng(42)
    values = np.zeros(800)
    values[200:] = rng.exponential(1.0, 600)  # 80% zeros + exponential tail
    params, _ = _fit_type_aware(values)
    assert params["anchor"] < 0.4, (
        f"Expected anchor < 0.4 for right-skewed data, got {params['anchor']:.3f}"
    )


def test_scaler_roundtrip_preserves_anchor(reference_df_vi, vector_index_dir, tmp_path):
    """save/load preserves anchor and use_log1p in scalers."""
    eq = ErsiliaQuality(k=10).fit(reference_df_vi, eos_id=EOS_ID, version=VERSION,
                                  vector_index=vector_index_dir, ignore_size=True)
    folder = tmp_path / "anchor_model"
    eq.save(folder)
    loaded = ErsiliaQuality.load(folder)
    orig_scalers = eq._fit_state.preprocess_state["scalers"]
    load_scalers = loaded._fit_state.preprocess_state["scalers"]
    for col in orig_scalers:
        assert orig_scalers[col]["anchor"] == load_scalers[col]["anchor"]
        assert orig_scalers[col]["use_log1p"] == load_scalers[col]["use_log1p"]
