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


def test_unparsable_smiles_give_nan(fitted):
    q = pd.DataFrame({"key": ["ok", "bad"], "input": ["CCO", "not a smiles"]})
    res = fitted.training_physchem.run(q)
    assert np.isfinite(res.score.iloc[0]) and np.isnan(res.score.iloc[1])
    assert np.isfinite(res.score_raw.iloc[0])


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
