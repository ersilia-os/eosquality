import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.exceptions import SchemaError
from eosquality.training import load_training

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
        eos_id="eos0aaa",
        vector_index=library,
        ignore_size=True,
        training_sets=training_dir,
    )


def test_training_molecules_get_their_loo_distance(both):
    distance = both.training_distance
    for name, column in distance.training_.columns.items():
        q = pd.DataFrame({"key": column.ids, "input": column.smiles})
        det = distance.run(q).details
        d = det[det.column == name]
        assert d.in_training.all()
        # Self excluded: the raw values are exactly the leave-one-out table.
        assert (d.distance_raw > 0).all()
        np.testing.assert_allclose(
            np.sort(d.distance_raw), distance._loo[name], atol=1e-6
        )
        # Training molecules against their own CDF average 0.5 (mid-ranks; the
        # 1/(2n) floor clip adds a hair).
        assert d.distance.mean() == pytest.approx(0.5, abs=1e-3)


def test_distance_matches_brute_force_tanimoto(both, query):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    column = both.training_distance.training_.columns["mw"]
    train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in column.smiles]
    det = both.run(query).training_details
    det = det[det.column == "mw"].set_index("key")
    for key, smi in zip(query.key[:10], query.input[:10], strict=True):
        if det.loc[key, "in_training"]:
            continue
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smi))
        sims = np.sort(DataStructs.BulkTanimotoSimilarity(fp, train_fps))[::-1]
        assert det.loc[key, "nn1_distance"] == pytest.approx(1 - sims[0], abs=1e-6)
        assert det.loc[key, "distance_raw"] == pytest.approx(
            1 - sims[:5].mean(), abs=1e-6
        )


def test_unrelated_molecule_is_far(both):
    q = pd.DataFrame({"key": ["far"], "input": ["[Fe+2].[Cl-].[Cl-]"]})
    res = both.training_distance.run(q)
    assert res.score.iloc[0] > 0.95
    assert res.score_raw.iloc[0] > 0.9


def test_summaries_are_q66_of_columns(both, query):
    res = both.run(query)
    for value, score in (
        ("distance", "training_distance"),
        ("distance_raw", "training_distance_raw"),
    ):
        per_col = res.training_details.pivot(
            index="key", columns="column", values=value
        )
        expected = per_col.quantile(0.66, axis=1).reindex(query.key).to_numpy()
        np.testing.assert_allclose(res.scores[score].to_numpy(), expected)


def test_run_columns_and_details(both, query):
    result = both.run(query)
    for c in (
        "training_distance",
        "training_distance_raw",
        "training_n_columns",
        "in_training_any",
    ):
        assert c in result.scores.columns
    assert "support" in result.scores.columns  # reference modality still there
    det = result.training_details
    assert len(det) == len(query) * 3
    assert set(det.column) == {"mw", "aromatic", "hbd"}
    row = det[det.column == "aromatic"].iloc[0]
    assert len(row.nn_keys.split("|")) == 5 and set(row.nn_y.split("|")) <= {"0", "1"}
    assert (det[det.column == "hbd"].nn_y == "").all()


def test_roundtrip_with_training(both, query, tmp_path):
    before = both.run(query)
    both.save(tmp_path / "art")
    loaded = ErsiliaQuality.load(tmp_path / "art")
    assert loaded.modalities_ == ["reference", "training"]
    after = loaded.run(query)
    pd.testing.assert_frame_equal(before.scores, after.scores)
    pd.testing.assert_frame_equal(before.training_details, after.training_details)


def test_training_only(training_dir, query, tmp_path):
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=training_dir)
    assert eq.modalities_ == ["training"]
    smiles_only = query[["key", "input"]]
    res = eq.run(smiles_only)
    assert list(res.scores.columns) == [
        "training_distance",
        "training_distance_raw",
        "training_n_columns",
        "in_training_any",
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
        reference, eos_id="eos0aaa", vector_index=library, ignore_size=True
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
    assert "training_distance" in loaded.run(query).scores.columns
    with pytest.raises(FileExistsError):
        ErsiliaQuality.add_training(art, training_dir)


def test_add_training_rejects_other_model(reference, library, training_dir, tmp_path):
    art = tmp_path / "art"
    ErsiliaQuality().fit(
        reference,
        eos_id="eos0aaa",
        vector_index=library,
        ignore_size=True,
        scores=["typicality"],
    ).save(art)
    with pytest.raises(ValueError, match="eos9zzz"):
        ErsiliaQuality.add_training(art, training_dir, eos_id="eos9zzz")


def test_training_files_named_after_ersilia_columns(tmp_path, smiles, make_outputs):
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
        eos_id="eos42ez",
        ignore_size=True,
        scores=["typicality"],
        training_sets=folder,
    )
    assert eq.training_distance.training_.column_names == columns
    details = eq.run(ref.head(5)).training_details
    assert sorted(details.column.unique()) == columns
    # A file that is not named after an output column is rejected.
    (folder / "cytotoxicity_hela.csv").write_text("smiles\nCCO\n")
    with pytest.raises(SchemaError, match="cytotoxicity_hela"):
        ErsiliaQuality().fit(
            ref,
            eos_id="eos42ez",
            ignore_size=True,
            scores=["typicality"],
            training_sets=folder,
        )
