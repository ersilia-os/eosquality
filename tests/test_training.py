import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.exceptions import SchemaError
from eosquality.scores._error_model import feature_names
from eosquality.training import load_training

# The error model's four inputs, emitted alongside the scores.
ERROR_MODEL_INPUTS = feature_names()
# The physchem applicability domain, calibrated and raw.
PHYSCHEM_COLUMNS = ["trn_physchem_pct", "trn_physchem_raw", "trn_physchem_dist"]
# The structural applicability domain, calibrated and raw.
TANIMOTO_COLUMNS = ["trn_tanimoto_pct", "trn_tanimoto_raw"]
# The exact-structure and scaffold flags.
MATCH_COLUMNS = ["trn_match", "trn_scaffold"]
# What a default fit writes to the scores CSV (the error model is off).
DEFAULT_COLUMNS = [
    *TANIMOTO_COLUMNS,
    "trn_physchem_pct",
    "trn_physchem_raw",
    *MATCH_COLUMNS,
]

# Exclude every reference score but typicality (fast reference fits).
ONLY_TYPICALITY = [
    "ref_extremity",
    "ref_support",
    "ref_consistency",
    "ref_signal",
]

COLUMNS = ["mw", "logp", "tpsa", "aromatic", "hbd", "noisy"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_kinds_order_and_merging(training_dir):
    cols = load_training(training_dir, COLUMNS)
    assert list(cols) == ["mw", "aromatic", "hbd"]  # schema order
    assert cols["mw"].y_kind == "continuous"
    assert cols["aromatic"].y_kind == "binary"
    assert cols["hbd"].y is None and cols["hbd"].y_kind is None
    # 300 rows + a salt form of row 0 + a duplicate of row 1 → 300 molecules.
    assert cols["mw"].n == 300
    assert len(set(cols["mw"].smiles)) == cols["mw"].n


def test_loader_rejects_unknown_column(tmp_path, training_dir):
    (tmp_path / "not_a_column.csv").write_text("smiles\nCCO\n")
    with pytest.raises(SchemaError):
        load_training(tmp_path, COLUMNS)


def test_loader_needs_smiles_column(tmp_path):
    (tmp_path / "mw.csv").write_text("molecule,y\nCCO,1\n")
    with pytest.raises(SchemaError):
        load_training(tmp_path, COLUMNS)


def test_loader_reads_value_as_label(tmp_path, training_dir):
    expected = load_training(training_dir, COLUMNS)["mw"]
    df = pd.read_csv(training_dir / "mw.csv")
    df.rename(columns={"y": "value"}).to_csv(tmp_path / "mw.csv", index=False)
    column = load_training(tmp_path, COLUMNS)["mw"]
    assert column.y_kind == expected.y_kind
    np.testing.assert_array_equal(column.y, expected.y)
    # With both, 'y' wins.
    df.assign(value=-1.0).to_csv(tmp_path / "mw.csv", index=False)
    np.testing.assert_array_equal(load_training(tmp_path, COLUMNS)["mw"].y, expected.y)


def test_loader_skips_small_columns(tmp_path, training_dir):
    for f in training_dir.glob("*.csv"):
        (tmp_path / f.name).write_text(f.read_text())
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"]}).to_csv(
        tmp_path / "logp.csv", index=False
    )
    assert "logp" not in load_training(tmp_path, COLUMNS)


def test_loader_without_schema_takes_all_files(training_dir):
    assert set(load_training(training_dir)) == {"mw", "aromatic", "hbd"}


# ---------------------------------------------------------------------------
# Domain score
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def both(reference, library, training_dir):
    return ErsiliaQuality().fit(
        reference,
        training_dir,
        eos_id="eos0aaa",
        vector_index=library,
        exclude=["ref_signal"],
    )


def test_training_molecules_get_their_loo_distance(both):
    distance = both.training_distance
    for j, (name, column) in enumerate(distance.training_.columns.items()):
        raw, calibrated, in_train, _ = distance._per_column(column.smiles)
        assert in_train[:, j].all()
        # Self excluded: the raw values are exactly the leave-one-out table.
        assert (raw[:, j] > 0).all()
        np.testing.assert_allclose(np.sort(raw[:, j]), distance._loo[name], atol=1e-6)
        # Training molecules against their own CDF average 0.5 (mid-ranks; the
        # 1/(2n) floor clip adds a hair).
        assert calibrated[:, j].mean() == pytest.approx(0.5, abs=1e-3)
    column = next(iter(distance.training_.columns.values()))
    q = pd.DataFrame({"key": column.ids, "input": column.smiles})
    assert distance.run(q).in_training.all()


def test_distance_matches_brute_force_tanimoto(both, query):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    distance = both.training_distance
    columns = list(distance.training_.columns.values())
    train_fps = {
        c.name: [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in c.smiles]
        for c in columns
    }
    smiles = list(query.input[:10])
    raw, _, in_train, _ = distance._per_column(smiles)
    det = both.run(query).training_details.set_index("key")
    for i, (key, smi) in enumerate(zip(query.key[:10], smiles, strict=True)):
        if in_train[i].any():
            continue
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smi))
        nearest = []
        for j, c in enumerate(columns):
            sims = np.sort(DataStructs.BulkTanimotoSimilarity(fp, train_fps[c.name]))
            assert raw[i, j] == pytest.approx(1 - sims[::-1][:5].mean(), abs=1e-6)
            nearest.append(sims[-1])
        assert det.loc[key, "nn1_similarity"] == pytest.approx(max(nearest), abs=1e-6)


def test_details_keep_unparsable_queries(both):
    q = pd.DataFrame({"key": ["ok", "bad"], "input": ["CCO", "not a smiles"]})
    det = both.training_distance.run(q).details
    assert det.key.tolist() == ["ok", "bad"]
    assert np.isfinite(det.trn_tanimoto_pct[0]) and np.isnan(det.trn_tanimoto_pct[1])
    assert det.nn_smiles.isna().tolist() == [False, True]


def test_unrelated_molecule_is_far(both):
    q = pd.DataFrame({"key": ["far"], "input": ["[Fe+2].[Cl-].[Cl-]"]})
    res = both.training_distance.run(q)
    assert res.score.iloc[0] < 0.05  # similarity percentile: far is near 0
    assert res.score_raw.iloc[0] < 0.1  # a similarity: nothing like the training set
    assert not res.in_training.iloc[0]


def test_whole_model_value_is_q66_of_columns(both, query):
    distance = both.training_distance
    raw, calibrated, _, _ = distance._per_column(list(query.input))
    res = both.run(query)
    np.testing.assert_allclose(
        res.scores["trn_tanimoto_pct"], 1 - np.quantile(calibrated, 0.66, axis=1)
    )
    np.testing.assert_allclose(
        res.scores["trn_tanimoto_raw"], 1 - np.quantile(raw, 0.66, axis=1)
    )


def test_run_columns_and_details(both, query):
    result = both.run(query)
    for c in DEFAULT_COLUMNS:
        assert c in result.scores.columns
    assert "trn_in_training" in result.training_details.columns
    assert "trn_difficulty" not in result.scores.columns  # off by default
    assert "ref_support" in result.scores.columns  # reference modality still there
    det = result.training_details
    assert len(det) == len(query) and det.key.tolist() == query.key.tolist()
    assert det.input.tolist() == query.input.tolist()
    np.testing.assert_allclose(det.trn_tanimoto_pct, result.scores["trn_tanimoto_pct"])
    known = {"mw", "aromatic", "hbd"}
    for row in det.itertuples():
        sims = [float(v) for v in row.nn_similarities.split("|")]
        assert len(sims) == 5 and sims == sorted(sims, reverse=True)
        assert row.nn1_similarity == pytest.approx(sims[0], abs=1e-3)
        assert len(row.nn_smiles.split("|")) == 5
        assert len(set(row.nn_keys.split("|"))) == 5  # deduplicated
        for cols in row.nn_columns.split("|"):
            assert set(cols.split(";")) <= known


def test_roundtrip_with_training(both, query, tmp_path):
    before = both.run(query)
    both.save(tmp_path / "art")
    loaded = ErsiliaQuality.load(tmp_path / "art")
    assert loaded.modalities_ == ["reference", "training"]
    after = loaded.run(query)
    pd.testing.assert_frame_equal(before.scores, after.scores)
    pd.testing.assert_frame_equal(before.training_details, after.training_details)


def test_training_only(training_dir, query, tmp_path):
    eq = ErsiliaQuality().fit(training_sets=training_dir, eos_id="eos0aaa")
    assert eq.modalities_ == ["training"]
    smiles_only = query[["key", "input"]]
    res = eq.run(smiles_only)
    assert list(res.scores.columns) == DEFAULT_COLUMNS
    eq.save(tmp_path / "art")
    assert not (tmp_path / "art/reference_mode").exists()
    assert (tmp_path / "art/training_mode/training_sets").is_dir()
    pd.testing.assert_frame_equal(
        ErsiliaQuality.load(tmp_path / "art").run(smiles_only).scores, res.scores
    )


def test_reference_is_restricted_to_training_columns(both, reference, library):
    trained = ["mw", "aromatic", "hbd"]  # the fixture's usable training files
    assert both.shared_.schema.column_names == trained
    assert set(both.shared_.selected_columns) <= set(trained)
    # Without training sets, the reference keeps every output column.
    alone = ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, exclude=ONLY_TYPICALITY
    )
    assert alone.shared_.schema.column_names == COLUMNS


def test_max_features_selects_within_training_columns(reference, library, training_dir):
    eq = ErsiliaQuality().fit(
        reference,
        training_dir,
        eos_id="eos0aaa",
        vector_index=library,
        exclude=ONLY_TYPICALITY,
        max_features=2,
    )
    assert eq.shared_.schema.column_names == ["mw", "aromatic", "hbd"]
    assert len(eq.shared_.selected_columns) == 2
    assert set(eq.shared_.selected_columns) <= {"mw", "aromatic", "hbd"}
    # The training modality uses the same columns.
    assert eq.training_distance.training_.column_names == eq.shared_.selected_columns


def test_training_only_keeps_least_overlapping_columns(training_dir):
    eq = ErsiliaQuality().fit(
        training_sets=training_dir, eos_id="eos0aaa", max_features=2
    )
    kept = eq.training_distance.training_.column_names
    assert len(kept) == 2 and set(kept) <= {"mw", "aromatic", "hbd"}


def test_select_by_shared_molecules_drops_a_panel_twin():
    from eosquality.shared.feature_selection import select_by_shared_molecules

    panel = {f"m{i}" for i in range(100)}
    sets = {
        "a": panel,
        "a_twin": panel | {"x"},  # the same screening panel
        "b": {f"n{i}" for i in range(100)},
    }
    kept = select_by_shared_molecules(sets, 2)
    assert "b" in kept and len(kept) == 2
    assert select_by_shared_molecules(sets, None) == ["a", "a_twin", "b"]


def test_select_by_shared_molecules_keeps_the_largest_of_a_panel():
    from eosquality.shared.feature_selection import select_by_shared_molecules

    panel = {f"m{i}" for i in range(100)}
    sets = {
        "small_twin": panel | {"x"},
        "big_twin": panel | {f"e{i}" for i in range(400)},
        "other": {f"n{i}" for i in range(50)},
    }
    assert select_by_shared_molecules(sets, 2) == ["big_twin", "other"]


def test_training_files_named_after_ersilia_columns(
    tmp_path, smiles, make_outputs, library
):
    """Training files are matched to model outputs by Ersilia column name."""
    columns = ["cytotoxicity_hepg2", "cytotoxicity_hskmc", "cytotoxicity_imr90"]
    ref = make_outputs(smiles[:600], seed=0)[["key", "input", "mw", "logp", "tpsa"]]
    ref.columns = ["key", "input", *columns]
    folder = tmp_path / "training_eos42ez_v1"
    folder.mkdir()
    for i, col in enumerate(columns):
        part = smiles[100 * i : 100 * i + 200]
        pd.DataFrame({"smiles": part, "y": np.arange(len(part)) % 2}).to_csv(
            folder / f"{col}.csv", index=False
        )
    eq = ErsiliaQuality().fit(
        ref,
        folder,
        eos_id="eos42ez",
        vector_index=library,
        exclude=ONLY_TYPICALITY,
    )
    assert eq.training_distance.training_.column_names == columns
    details = eq.run(ref.head(5)).training_details
    assert len(details) == 5
    member_of = {
        c for cell in details.nn_columns for n in cell.split("|") for c in n.split(";")
    }
    assert member_of <= set(columns)
    # A file that is not named after an output column is rejected.
    (folder / "cytotoxicity_hela.csv").write_text("smiles\nCCO\n")
    with pytest.raises(SchemaError, match="cytotoxicity_hela"):
        ErsiliaQuality().fit(
            ref,
            folder,
            eos_id="eos42ez",
            vector_index=library,
            exclude=ONLY_TYPICALITY,
        )


def test_scaffold_survives_rdkit_stereo_failure():
    from eosquality.training.folds import _scaffold, scaffold_folds

    # RDKit cannot canonicalise this scaffold with its stereo double bonds.
    tricky = "N#C/C(=C\\C=C\\c1ccccc1)c1ccc(F)cc1"
    assert _scaffold(tricky) == "C(C=Cc1ccccc1)=Cc1ccccc1"
    assert len(scaffold_folds([tricky, "CCO", "c1ccccc1C"], n_folds=2)) == 3


def test_excluding_one_training_score(training_dir, query):
    """Any training score can be fitted alone; the details file follows."""
    smiles_only = query[["key", "input"]]
    no_tanimoto = ErsiliaQuality().fit(
        training_sets=training_dir, eos_id="eos0aaa", exclude=["trn_tanimoto"]
    )
    res = no_tanimoto.run(smiles_only)
    assert list(res.scores.columns) == [
        "trn_physchem_pct",
        "trn_physchem_raw",
        *MATCH_COLUMNS,
    ]
    # Without trn_tanimoto the details table is just key, input and the rest.
    assert list(res.training_details.columns) == [
        "key",
        "input",
        *PHYSCHEM_COLUMNS,
        *MATCH_COLUMNS,
    ]

    only_match = ErsiliaQuality().fit(
        training_sets=training_dir,
        eos_id="eos0aaa",
        exclude=["trn_tanimoto", "trn_physchem"],
    )
    assert list(only_match.run(smiles_only).scores.columns) == MATCH_COLUMNS

    with pytest.raises(ValueError, match="nothing to fit"):
        ErsiliaQuality().fit(
            training_sets=training_dir,
            eos_id="eos0aaa",
            exclude=["trn_tanimoto", "trn_physchem", "trn_match"],
        )


def test_physchem_distance_is_only_in_the_details_file(both, query):
    from eosquality.scores._physchem_domain import PAIR_MEDIAN

    res = both.run(query)
    assert "trn_physchem_dist" not in res.scores.columns
    det = res.training_details
    np.testing.assert_allclose(det.trn_physchem_pct, res.scores.trn_physchem_pct)
    # The scores column is the similarity of the details file's distance.
    np.testing.assert_allclose(
        res.scores.trn_physchem_raw, 1 - det.trn_physchem_dist / PAIR_MEDIAN
    )


def test_error_model_is_off_unless_included(training_dir, query):
    smiles_only = query[["key", "input"]]
    default = ErsiliaQuality().fit(training_sets=training_dir, eos_id="eos0aaa")
    assert default.training_difficulty is None
    on = ErsiliaQuality().fit(
        training_sets=training_dir, eos_id="eos0aaa", include=["trn_difficulty"]
    )
    res = on.run(smiles_only)
    assert list(res.scores.columns) == [*DEFAULT_COLUMNS, "trn_difficulty"]
    det = res.training_details.columns
    assert all(f"trn_{f}" in det for f in ERROR_MODEL_INPUTS)


# ---------------------------------------------------------------- trn_match


def test_match_is_one_for_a_training_molecule_and_zero_for_a_stranger(both):
    known = both._training.columns["mw"].smiles[0]
    q = pd.DataFrame({"key": ["in", "out"], "input": [known, "[Fe+2].[Cl-].[Cl-]"]})
    res = both.training_match.run(q)
    assert res.match.tolist() == [1, 0]


def test_match_ignores_stereochemistry(training_dir):
    """The connectivity layer is blind to stereo, so enantiomers match."""
    eq = ErsiliaQuality().fit(training_sets=training_dir, eos_id="eos0aaa")
    layers = eq.training_match._molecules
    from eosquality.scores.training_match import connectivity_layer

    assert connectivity_layer("C[C@H](N)C(=O)O") == connectivity_layer(
        "C[C@@H](N)C(=O)O"
    )
    assert len(layers) and all(len(x) == 14 for x in layers)


def test_scaffold_is_missing_without_one_and_match_is_missing_if_unparsable(both):
    q = pd.DataFrame(
        {"key": ["ring", "chain", "bad"], "input": ["c1ccccc1CC", "CCCCO", "no"]}
    )
    res = both.training_match.run(q)
    assert pd.isna(res.scaffold.iloc[1]) and res.scaffold.iloc[0] in (0, 1)
    assert pd.isna(res.match.iloc[2]) and pd.isna(res.scaffold.iloc[2])
    assert str(res.match.dtype) == "Int64"


def test_scaffold_matches_through_the_ring_system(both):
    """A query that is a training molecule's bare scaffold matches on scaffold."""
    from eosquality.training.folds import _scaffold

    known = next(
        s for s in both._training.columns["mw"].smiles if _scaffold(s) not in ("", s)
    )
    q = pd.DataFrame({"key": ["scaffold"], "input": [_scaffold(known)]})
    assert both.training_match.run(q).scaffold.tolist() == [1]


def test_match_roundtrip(both, query, tmp_path):
    before = both.training_match.run(query)
    both.save(tmp_path / "art")
    after = ErsiliaQuality.load(tmp_path / "art").training_match.run(query)
    pd.testing.assert_series_equal(before.match, after.match)
    pd.testing.assert_series_equal(before.scaffold, after.scaffold)
