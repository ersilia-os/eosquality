import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from eosquality import ErsiliaQuality
from eosquality.exceptions import ArtifactVersionError
from eosquality.scores._error_model import MIN_LABELLED
from eosquality.training.folds import scaffold_folds


@pytest.fixture(scope="module")
def fitted(training_dir):
    return ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=training_dir)


def test_fitted_only_on_labelled_columns(fitted):
    models = fitted.training_difficulty.models_
    assert set(models) == {"mw", "aromatic"}  # hbd has no y
    assert not models["mw"].binary and models["aromatic"].binary
    for m in models.values():
        assert np.isfinite(m.spearman)
        assert m.n_labelled >= MIN_LABELLED


def test_scores_and_metadata(fitted, query):
    res = fitted.run(query[["key", "input"]])
    score = res.scores["trn_difficulty"]
    assert ((score > 0) & (score <= 1)).all()
    assert set(res.metadata["trn_difficulty_spearman"]) == {"mw", "aromatic"}
    details = res.training_details
    assert list(details.columns[2:6]) == [
        "trn_distance",
        "trn_distance_raw",
        "trn_difficulty",
        "trn_in_training",
    ]
    np.testing.assert_allclose(details.trn_difficulty, score)


def test_roundtrip(fitted, query, tmp_path):
    q = query[["key", "input"]]
    before = fitted.run(q).scores
    fitted.save(tmp_path / "art")
    assert (tmp_path / "art/training_mode/training_difficulty/c000").is_dir()
    after = ErsiliaQuality.load(tmp_path / "art").run(q).scores
    pd.testing.assert_frame_equal(before, after)


def test_sklearn_version_mismatch_refuses_to_load(fitted, tmp_path):
    fitted.save(tmp_path / "art")
    state = tmp_path / "art/training_mode/training_difficulty/c000/state.json"
    state.write_text(
        state.read_text().replace('"sklearn_version": "', '"sklearn_version": "0.')
    )
    with pytest.raises(ArtifactVersionError, match="scikit-learn"):
        ErsiliaQuality.load(tmp_path / "art")


def test_no_labels_no_difficulty(tmp_path, smiles):
    folder = tmp_path / "training_eos0aaa_v1"
    folder.mkdir()
    pd.DataFrame({"smiles": smiles[:200]}).to_csv(folder / "mw.csv", index=False)
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=folder)
    assert eq.training_difficulty is None
    assert "trn_difficulty" not in eq.run(pd.DataFrame({"input": smiles[:5]})).scores


def test_noisy_region_ranks_harder(tmp_path, smiles):
    """Clean heavy-atom-count labels, except pure noise for S-containing molecules."""
    rng = np.random.default_rng(0)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    noisy = np.array([any(a.GetSymbol() == "S" for a in m.GetAtoms()) for m in mols])
    y = np.array([m.GetNumHeavyAtoms() for m in mols], dtype=float)
    y[noisy] = rng.normal(20, 10, noisy.sum())
    folder = tmp_path / "training_eos0aaa_v1"
    folder.mkdir()
    pd.DataFrame({"smiles": smiles[:600], "y": y[:600]}).to_csv(
        folder / "mw.csv", index=False
    )
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=folder)
    # The four inputs say how far and how uncertain, never which chemotype, so
    # a noisy region defined by a substructure is only partly recoverable: the
    # surrogate's own variance rises there, but nothing names sulfur. With the
    # earlier 2048-bit structural inputs this reached > 0.2.
    assert eq.training_difficulty.models_["mw"].spearman > 0.05
    score = eq.run(pd.DataFrame({"input": smiles[600:]})).scores["trn_difficulty"]
    held_out_noisy = noisy[600:]
    assert score[held_out_noisy].mean() > score[~held_out_noisy].mean() + 0.1


def test_scaffold_folds_keep_scaffolds_together(smiles):
    from rdkit.Chem.Scaffolds import MurckoScaffold

    folds = scaffold_folds(smiles[:300], n_folds=5)
    assert set(folds) == set(range(5))
    by_scaffold = {}
    for s, f in zip(smiles[:300], folds, strict=True):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        if scaffold:
            by_scaffold.setdefault(scaffold, set()).add(f)
    assert all(len(v) == 1 for v in by_scaffold.values())


def test_out_of_fold_spearman_is_recorded(fitted):
    for m in fitted.training_difficulty.models_.values():
        assert np.isfinite(m.spearman)
        assert m.error_model.n_features_in_ == len(m.features)


def test_inputs_are_the_four_scalars(fitted):
    """Two neighbour similarities, the ensemble variance and the score."""
    from eosquality.scores import _error_model as em

    model = fitted.training_difficulty.models_["aromatic"]
    names = em.feature_names()
    assert names == [
        "nn1_tanimoto",
        "nn5_tanimoto",
        "ensemble_variance",
        "surrogate_score",
    ]
    n = 7
    p = model.oof_prediction[:n]
    dist = np.linspace(0.1, 0.9, n * model.k).reshape(n, model.k)
    inputs = em._inputs(dist=dist, prediction=p, variance=model.oof_variance[:n])
    assert inputs.shape == (n, len(names))
    # Distances in, similarities out.
    np.testing.assert_allclose(inputs[:, 0], 1 - dist[:, 0])
    np.testing.assert_allclose(inputs[:, 1], 1 - dist.mean(axis=1))
    np.testing.assert_allclose(inputs[:, 3], p)


def test_surrogate_is_class_weighted_for_binary_labels():
    from eosquality.scores._error_model import _new_surrogate

    assert _new_surrogate(binary=True).class_weight == "balanced"
    # sklearn's regressor carries the attribute but ignores it; it stays unset.
    assert _new_surrogate(binary=False).class_weight is None


def test_congeneric_series_falls_back_to_random_folds(tmp_path):
    from eosquality.training.folds import cv_folds

    subs = ["C", "CC", "Cl", "F", "Br", "O", "N", "OC", "C#N", "CO", "S", "I"]
    smi = sorted({f"c1cc({a})ccc1{b}" for a in subs for b in subs})[:80]
    folds, kind = cv_folds(smi, np.ones(len(smi), dtype=bool))
    assert kind == "random" and len(set(folds)) == 5
    folder = tmp_path / "training_eos0aaa_v1"
    folder.mkdir()
    y = np.random.default_rng(0).normal(size=len(smi))
    pd.DataFrame({"smiles": smi, "y": y}).to_csv(folder / "mw.csv", index=False)
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=folder)  # no crash
    assert eq.training_difficulty.models_["mw"].cv == "random"


def test_training_molecules_get_their_out_of_fold_difficulty(fitted):
    from eosquality.scores._helpers import _cdf_score

    training = fitted._training
    column, vi = training.columns["mw"], training.indices["mw"]
    model = fitted.training_difficulty.models_["mw"]
    predicted = model.predict(column, vi, column.smiles[:25])
    np.testing.assert_allclose(predicted, model.oof_error[:25])
    calibrated = _cdf_score(
        model.oof_error, model.sorted_oof_error, higher_is_higher=True
    )
    assert np.nanmean(calibrated) == pytest.approx(0.5, abs=0.01)


def test_error_models_fit_on_at_most_max_fit_molecules(training_dir, monkeypatch):
    from eosquality.scores import _error_model

    monkeypatch.setattr(_error_model, "MAX_FIT_MOLECULES", 120)
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=training_dir)
    model = eq.training_difficulty.models_["mw"]
    column, vi = eq._training.columns["mw"], eq._training.indices["mw"]
    assert model.n_fit == 120 and model.n_labelled == column.n > 120
    assert np.isfinite(model.oof_error).sum() == 120
    assert len(model.sorted_oof_error) == 120
    # Training molecules outside the fitted subset are scored like queries.
    predicted = model.predict(column, vi, column.smiles)
    assert np.isfinite(predicted).all()
    fitted_rows = np.isfinite(model.oof_error)
    np.testing.assert_allclose(predicted[fitted_rows], model.oof_error[fitted_rows])


def test_weak_error_model_is_reported(training_dir, tmp_path, monkeypatch):
    from eosquality.scores import training_difficulty as td
    from eosquality.utils.logging import logger

    monkeypatch.setattr(td, "WEAK_SPEARMAN", 0.999)  # every model counts as weak
    path = tmp_path / "fit.log"
    with logger.log_file(path):
        ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=training_dir)
    assert "barely predictable" in path.read_text()
