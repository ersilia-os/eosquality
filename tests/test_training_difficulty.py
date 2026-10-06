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
    score = res.scores["training_difficulty"]
    assert ((score > 0) & (score <= 1)).all()
    assert set(res.metadata["training_difficulty_spearman"]) == {"mw", "aromatic"}
    assert "difficulty" in res.training_details.columns
    np.testing.assert_allclose(res.training_details.difficulty, score)


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
    assert (
        "training_difficulty" not in eq.run(pd.DataFrame({"input": smiles[:5]})).scores
    )


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
    assert eq.training_difficulty.models_["mw"].spearman > 0.2
    score = eq.run(pd.DataFrame({"input": smiles[600:]})).scores["training_difficulty"]
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


def test_inputs_follow_unique_feature_set_one(fitted):
    from eosquality.scores import _error_model as em

    model = fitted.training_difficulty.models_["aromatic"]
    names = em.feature_names(binary=True)
    assert names[:166] == list(em.MACCS_NAMES) and names[-1] == "prediction"
    assert "probability_top1" in names and "probability_top1" not in (
        em.feature_names(binary=False)
    )
    n = 7
    p = model.oof_prediction[:n]
    inputs = em._inputs(
        dist=np.zeros((n, model.k)),
        prediction=p,
        variance=model.oof_variance[:n],
        log_density=np.zeros((n, 3)),
        maccs=np.zeros((n, 166)),
        binary=True,
    )
    assert inputs.shape == (n, len(names))
    top1 = inputs[:, names.index("probability_top1")]
    np.testing.assert_allclose(top1, np.maximum(p, 1 - p))
    assert ((top1 >= 0.5) & (top1 <= 1.0)).all()


def test_kde_matches_brute_force(smiles):
    from scipy.special import logsumexp
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import KernelDensity

    from eosquality.library.maccs import compute_maccs
    from eosquality.scores._density import KDE_VARIANTS, TrainingDensity

    maccs = compute_maccs(smiles[:120], show_progress=False).astype(float)
    density = TrainingDensity.fit(maccs)
    new = compute_maccs(smiles[600:603], show_progress=False).astype(float)
    loo = density.log_density(maccs[:3], np.arange(3))
    plain = density.log_density(new, np.full(3, -1))
    for j, (kernel, metric) in enumerate(KDE_VARIANTS):
        h = density.bandwidths[j]
        point = maccs[:1]
        single = KernelDensity(kernel=kernel, metric=metric, bandwidth=h)
        log_k0 = single.fit(point).score_samples(point)[0]

        def brute(q, ref, kernel=kernel, metric=metric, h=h, log_k0=log_k0):
            d = pairwise_distances(q, ref, metric=metric)
            log_k = log_k0 - (d * d / (2 * h * h) if kernel == "gaussian" else d / h)
            return logsumexp(log_k, axis=1) - np.log(len(ref))

        for i in range(3):
            rest = np.delete(maccs, i, axis=0)
            assert loo[i, j] == pytest.approx(brute(maccs[i : i + 1], rest)[0])
        np.testing.assert_allclose(plain[:, j], brute(new, maccs))


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
