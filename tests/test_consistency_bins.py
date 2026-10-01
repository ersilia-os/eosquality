import numpy as np

from eosquality.scores.consistency import (
    _assign_fp_bins,
    _compute_fp_bin_edges,
    _consistency_from_distances_binned,
    _partition_and_sort,
)


def test_duplicate_quantiles_collapse():
    # 90% of rows share one value: most decile edges coincide.
    x = np.concatenate([np.full(900, 0.5), np.linspace(0.6, 0.9, 100)])
    edges = _compute_fp_bin_edges(x, 10)
    assert np.all(np.diff(edges) > 0)
    counts = np.bincount(_assign_fp_bins(x, edges))
    assert (counts > 0).all()


def test_small_bins_are_merged():
    x = np.random.default_rng(0).uniform(size=1000)
    edges = _compute_fp_bin_edges(x, 10, min_bin_size=250)
    counts = np.bincount(_assign_fp_bins(x, edges), minlength=len(edges) - 1)
    assert counts.min() >= 250
    assert len(edges) - 1 < 10


def test_reference_is_calibrated_within_each_bin():
    rng = np.random.default_rng(1)
    fp = rng.uniform(size=5000)
    out = fp + rng.normal(0, 0.1, size=5000)  # output noise grows with FP distance
    edges = _compute_fp_bin_edges(fp, 5)
    per_bin = _partition_and_sort(out, fp, edges)
    score = _consistency_from_distances_binned(out, fp, edges, per_bin)
    for b in range(len(per_bin)):
        assert abs(score[_assign_fp_bins(fp, edges) == b].mean() - 0.5) < 1e-9


def test_nan_distances_are_excluded_and_score_nan():
    fp = np.linspace(0, 1, 100)
    out = fp.copy()
    out[::10] = np.nan
    edges = _compute_fp_bin_edges(fp, 2)
    per_bin = _partition_and_sort(out, fp, edges)
    assert all(np.isfinite(b).all() for b in per_bin)
    score = _consistency_from_distances_binned(out, fp, edges, per_bin)
    assert np.isnan(score[::10]).all()
    assert np.isfinite(np.delete(score, np.arange(0, 100, 10))).all()
