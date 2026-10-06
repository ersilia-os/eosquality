import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.exceptions import SchemaError
from eosquality.training import load_training

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
        assert det.loc[key, "nn1_distance"] == pytest.approx(1 - max(nearest), abs=1e-6)


def test_unrelated_molecule_is_far(both):
    q = pd.DataFrame({"key": ["far"], "input": ["[Fe+2].[Cl-].[Cl-]"]})
    res = both.training_distance.run(q)
    assert res.score.iloc[0] > 0.95
    assert res.score_raw.iloc[0] > 0.9
    assert not res.in_training.iloc[0]


def test_whole_model_value_is_q66_of_columns(both, query):
    distance = both.training_distance
    raw, calibrated, _, _ = distance._per_column(list(query.input))
    res = both.run(query)
    np.testing.assert_allclose(
        res.scores["trn_distance"], np.quantile(calibrated, 0.66, axis=1)
    )
    np.testing.assert_allclose(
        res.scores["trn_distance_raw"], np.quantile(raw, 0.66, axis=1)
    )


def test_run_columns_and_details(both, query):
    result = both.run(query)
    for c in ("trn_distance", "trn_distance_raw", "trn_in_training"):
        assert c in result.scores.columns
    assert "ref_support" in result.scores.columns  # reference modality still there
    det = result.training_details
    assert len(det) == len(query) and det.key.tolist() == query.key.tolist()
    np.testing.assert_allclose(det.distance, result.scores["trn_distance"])
    known = {"mw", "aromatic", "hbd"}
    for row in det.itertuples():
        sims = [float(v) for v in row.nn_similarities.split("|")]
        assert len(sims) == 5 and sims == sorted(sims, reverse=True)
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
    assert list(res.scores.columns) == [
        "trn_distance",
        "trn_distance_raw",
        "trn_difficulty",
        "trn_in_training",
    ]
    eq.save(tmp_path / "art")
    assert not (tmp_path / "art/reference_mode").exists()
    assert (tmp_path / "art/training_mode/training_sets").is_dir()
    pd.testing.assert_frame_equal(
        ErsiliaQuality.load(tmp_path / "art").run(smiles_only).scores, res.scores
    )


def _digest(folder):
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()


def test_add_training_in_place(reference, library, training_dir, query, tmp_path):
    art = tmp_path / "art"
    ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, exclude=["ref_signal"]
    ).save(art)
    shared_before = _digest(art / "reference_mode")
    ErsiliaQuality.add_training(art, training_dir, eos_id="eos0aaa")
    assert _digest(art / "reference_mode") == shared_before
    loaded = ErsiliaQuality.load(art)
    assert loaded.modalities_ == ["reference", "training"]
    assert json.loads((art / "manifest.json").read_text())["modalities"] == [
        "reference",
        "training",
    ]
    assert "trn_distance" in loaded.run(query).scores.columns
    with pytest.raises(FileExistsError):
        ErsiliaQuality.add_training(art, training_dir)


def test_add_training_rejects_other_model(reference, library, training_dir, tmp_path):
    art = tmp_path / "art"
    ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        exclude=ONLY_TYPICALITY,
    ).save(art)
    with pytest.raises(ValueError, match="eos9zzz"):
        ErsiliaQuality.add_training(art, training_dir, eos_id="eos9zzz")


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
