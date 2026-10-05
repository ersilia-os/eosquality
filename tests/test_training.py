import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

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
        training=training_dir,
    )


def test_training_molecules_score_uniform(both):
    domain = both.training_domain
    for name, column in domain.training_.columns.items():
        q = pd.DataFrame({"key": column.ids, "input": column.smiles})
        details = domain.run(q).details
        d = details[details.column == name]
        assert d.in_training.all()
        assert d.domain.mean() == pytest.approx(0.5, abs=0.02)
        assert kstest(d.domain, "uniform").statistic < 1.36 / np.sqrt(len(d))
        loo = 1 - domain.training_.indices[name].self_knn_distances(1)[:, 0]
        np.testing.assert_allclose(np.sort(d.domain_raw), np.sort(loo), atol=1e-6)


def test_run_columns_and_details(both, query):
    result = both.run(query)
    for c in (
        "training_domain",
        "training_domain_raw",
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
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training=training_dir)
    assert eq.modalities_ == ["training"]
    smiles_only = query[["key", "input"]]
    res = eq.run(smiles_only)
    assert list(res.scores.columns) == [
        "training_domain",
        "training_domain_raw",
        "training_n_columns",
        "in_training_any",
    ]
    eq.save(tmp_path / "art")
    assert not (tmp_path / "art/shared").exists()
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
    shared_before = _digest(art / "shared")
    ErsiliaQuality.add_training(art, training_dir, eos_id="eos0aaa")
    assert _digest(art / "shared") == shared_before
    loaded = ErsiliaQuality.load(art)
    assert loaded.modalities_ == ["reference", "training"]
    assert json.loads((art / "manifest.json").read_text())["modalities"] == [
        "reference",
        "training",
    ]
    assert "training_domain" in loaded.run(query).scores.columns
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
