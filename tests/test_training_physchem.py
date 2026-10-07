import warnings

import numpy as np
import pandas as pd
import pytest

from eosquality import ErsiliaQuality
from eosquality.scores._physchem_domain import CLIP, PhyschemDomain


@pytest.fixture(scope="module")
def fitted(training_dir):
    return ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=training_dir)


def test_domain_is_fitted_per_column(fitted):
    domains = fitted.training_physchem.domains_
    assert set(domains) == {"mw", "aromatic", "hbd"}
    for d in domains.values():
        assert d.k == 5
        assert d.n_train == len(d.sorted_distances)
        assert d.train.shape[1] == len(d.mean) == len(d.scale) == len(d.median)


def test_calibration_puts_training_molecules_near_a_half(fitted):
    """Per column, a typical training molecule sits at ~0.5 by construction.

    The whole-model score is the Q66 across columns, so it sits higher; the
    guarantee is per column, which is what this checks.
    """
    from eosquality.library.physchem import compute_physchem_raw
    from eosquality.scores._helpers import _cdf_score

    column = fitted._training.columns["mw"]
    domain = fitted.training_physchem.domains_["mw"]
    calibrated = _cdf_score(
        domain.measure(
            compute_physchem_raw(column.smiles), column.rows_of(column.smiles)
        ),
        domain.sorted_distances,
        higher_is_higher=True,
    )
    assert calibrated.mean() == pytest.approx(0.5, abs=0.05)
    assert ((calibrated > 0) & (calibrated <= 1)).all()


def _domain_on_random_descriptors(n=40, k=3, seed=0):
    from eosquality.library.physchem import canonical_scaler

    scaler = canonical_scaler()
    rng = np.random.default_rng(seed)
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    raw = mean + scale * rng.normal(size=(n, len(mean)))
    return raw, PhyschemDomain.fit(raw, scaler, k=k)


def test_leave_one_out_is_the_mean_distance_to_the_k_nearest_others():
    """Checked against brute force; the self distance is float noise, not 0."""
    raw, domain = _domain_on_random_descriptors()
    z = domain.train.astype(np.float64)
    pairwise = np.linalg.norm(z[:, None] - z[None], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    expected = np.sort(pairwise, axis=1)[:, : domain.k].mean(axis=1)
    measured = domain.measure(raw, np.arange(len(raw)))
    np.testing.assert_allclose(measured, expected, rtol=1e-4)
    np.testing.assert_allclose(domain.sorted_distances, np.sort(expected), rtol=1e-4)


def test_only_the_query_itself_is_left_out():
    """A twin with identical descriptors (an enantiomer, say) is a neighbour."""
    raw, domain = _domain_on_random_descriptors()
    twinned = np.vstack([raw, raw[:1]])  # row 40 duplicates row 0
    from eosquality.library.physchem import canonical_scaler

    domain = PhyschemDomain.fit(twinned, canonical_scaler(), k=3)
    not_a_member = domain.measure(raw[:1])  # kept: both zero-distance rows count
    member = domain.measure(raw[:1], np.array([0]))  # drops only itself
    assert member[0] > not_a_member[0]
    z = domain.train.astype(np.float64)
    others = np.sort(np.linalg.norm(z - z[0], axis=1))[1:]  # [twin 0, then rest]
    assert others[0] < 1e-3
    np.testing.assert_allclose(member[0], others[:3].mean(), rtol=1e-4)


def test_rows_of_finds_training_molecules(fitted):
    column = fitted._training.columns["mw"]
    rows = column.rows_of([column.smiles[3], "not in the training set"])
    assert rows.tolist() == [3, -1]


def test_an_extreme_molecule_is_far_out(fitted):
    """A tiny inorganic salt sits outside drug-like physchem space."""
    q = pd.DataFrame({"key": ["far"], "input": ["[Fe+2].[Cl-].[Cl-]"]})
    # Similarity percentile: far from the training set is near 0.
    assert fitted.training_physchem.run(q).score.iloc[0] < 0.1


def test_unparsable_smiles_give_nan(fitted):
    q = pd.DataFrame({"key": ["ok", "bad"], "input": ["CCO", "not a smiles"]})
    res = fitted.training_physchem.run(q)
    assert np.isfinite(res.score.iloc[0]) and np.isnan(res.score.iloc[1])
    assert np.isfinite(res.score_raw.iloc[0])


def test_run_columns_are_in_range(fitted, query):
    res = fitted.training_physchem.run(query)
    assert ((res.score > 0) & (res.score <= 1)).all()
    assert (res.score_raw <= 1).all()  # a similarity: 1 is identical
    assert (res.distance > 0).all()


def test_roundtrip(fitted, query, tmp_path):
    before = fitted.training_physchem.run(query)
    fitted.save(tmp_path / "art")
    after = ErsiliaQuality.load(tmp_path / "art").training_physchem.run(query)
    pd.testing.assert_series_equal(before.score, after.score)
    pd.testing.assert_series_equal(before.score_raw, after.score_raw)


def _scaler(raw):
    """Library-style scaler parameters fitted on a synthetic matrix."""
    finite = np.where(np.isfinite(raw), raw, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        median = np.nan_to_num(np.nanmedian(finite, axis=0))
        mean = np.nan_to_num(np.nanmean(finite, axis=0))
        scale = np.nan_to_num(np.nanstd(finite, axis=0))
    return {"median": median, "mean": mean, "scale": scale}


def test_domain_handles_constant_and_non_finite_descriptors():
    """Zero-variance and NaN columns must not produce NaN distances."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(60, 12))
    raw[:, 3] = 2.5  # constant
    raw[:, 7] = np.nan  # never finite
    raw[0, 1] = np.inf  # a single bad cell
    domain = PhyschemDomain.fit(raw, _scaler(raw))
    assert np.isfinite(domain.measure(raw[:5])).all()
    assert np.isfinite(domain.sorted_distances).all()


def test_standardisation_stops_one_descriptor_dominating():
    """Without scaling, a large-magnitude descriptor would swamp the distance."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(200, 4))
    raw[:, 0] *= 10_000  # e.g. molecular weight against a 0-1 fraction
    domain = PhyschemDomain.fit(raw, _scaler(raw))
    assert domain.scale[0] > 100 * domain.scale[1]
    # After standardising, every descriptor has unit spread.
    spread = domain.train.astype(np.float64).std(axis=0)
    assert np.allclose(spread, 1.0, atol=0.05)


def test_scaled_values_are_clipped():
    """One exponential descriptor (like Ipc) must not swamp the distance."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(100, 6))
    scaler = _scaler(raw)
    domain = PhyschemDomain.fit(raw, scaler)
    outlier = raw[:1].copy()
    outlier[0, 2] = 1e30
    assert np.abs(domain.train).max() <= CLIP
    far = domain.measure(outlier)[0]
    # Clipped, no pair of points can be further apart than the cube's diagonal.
    assert np.isfinite(far) and far <= np.sqrt(6) * 2 * CLIP


def test_scaler_must_match_the_descriptor_count():
    raw = np.random.default_rng(0).normal(size=(30, 5))
    scaler = _scaler(raw[:, :4])
    with pytest.raises(ValueError, match="4 descriptors, the matrix 5"):
        PhyschemDomain.fit(raw, scaler)


def test_the_packaged_scaler_is_the_canonical_librarys():
    """The shipped copy must equal what ``eosquality build`` wrote."""
    import glob
    import json

    from eosquality.library.physchem import canonical_scaler

    found = glob.glob("data/indices/*/physchem_scaler.json")
    if not found:
        pytest.skip("canonical library not installed here")
    with open(found[0]) as f:
        library = json.load(f)
    shipped = canonical_scaler()
    for key in ("descriptor_names", "median", "mean", "scale"):
        assert shipped[key] == library[key], key


def test_similarity_is_one_minus_distance_over_the_pair_median():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(60, 4))
    domain = PhyschemDomain.fit(raw, _scaler(raw), pair_median=10.0)
    np.testing.assert_allclose(domain.similarity([0.0, 5.0, 10.0]), [1.0, 0.5, 0.0])


def test_similarity_is_not_clipped_at_zero(fitted):
    """Further than a random library pair gives a negative similarity."""
    q = pd.DataFrame({"key": ["far"], "input": ["[Fe+2].[Cl-].[Cl-]"]})
    res = fitted.training_physchem.run(q)
    pair_median = next(iter(fitted.training_physchem.domains_.values())).pair_median
    assert res.score_raw.iloc[0] == pytest.approx(
        1 - res.distance.iloc[0] / pair_median
    )
    assert res.score_raw.iloc[0] < 0
