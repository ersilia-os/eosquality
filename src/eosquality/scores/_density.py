"""Kernel densities of a training set, inputs of the training error model.

Follows UNIQUE's data-based KDE methods (``unique.uncertainty.data_based.
kernel_density``): three scikit-learn ``KernelDensity`` estimators on the
data features (here the MACCS keys), Gaussian/Euclidean, Gaussian/Manhattan
and exponential/Manhattan, each with its bandwidth chosen by
``GridSearchCV`` over ``{0.1, 0.5, 1}`` (5-fold, held-out log-likelihood).
The output is the log-density.

Two departures, both for large training sets or for fairness to training
molecules:

- **Size caps.** The grid search uses at most ``KDE_GRID_MAX`` molecules and
  the fitted KDE keeps at most ``KDE_REFERENCE_MAX`` (seeded random subsets),
  because a KDE costs O(n²) in the number of molecules. Smaller sets are used
  whole, as in UNIQUE.
- **Leave-one-out.** A molecule that is in the KDE's reference set has its
  own kernel removed (UNIQUE keeps it), so training molecules are scored like
  new molecules. The remaining kernels are summed exactly in log space;
  ``log K(0)`` comes from a KDE fitted on a single point, which carries
  scikit-learn's normalisation for every kernel and metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

KDE_VARIANTS = (
    ("gaussian", "euclidean"),
    ("gaussian", "manhattan"),
    ("exponential", "manhattan"),
)
KDE_NAMES = tuple(f"kde_{kernel}_{metric}" for kernel, metric in KDE_VARIANTS)
BANDWIDTHS = (0.1, 0.5, 1.0)
GRID_CV = 5
KDE_GRID_MAX = 2000
KDE_REFERENCE_MAX = 5000
# Query rows per pairwise-distance block.
_CHUNK = 1024


@dataclass
class TrainingDensity:
    """Three KDEs over (a subset of) a training set's MACCS keys."""

    bandwidths: np.ndarray  # (3,) grid-searched bandwidth of each KDE
    log_k0: np.ndarray  # (3,) log of each normalised kernel at distance 0
    reference_rows: np.ndarray  # training-set rows the densities are built on
    reference_maccs: np.ndarray  # (n_ref, 166) uint8 MACCS keys of those rows

    @classmethod
    def fit(cls, maccs: np.ndarray, seed: int = 0) -> TrainingDensity:
        """Grid-search each KDE's bandwidth, as in UNIQUE.

        Parameters
        ----------
        maccs : numpy.ndarray
            ``(n, 166)`` MACCS keys of the training molecules.
        seed : int, optional
            Seed for the subsets of large training sets.

        Returns
        -------
        TrainingDensity
        """
        x = maccs.astype(np.float64)
        rng = np.random.default_rng(seed)
        grid_rows = _subset(len(x), KDE_GRID_MAX, rng)
        reference_rows = _subset(len(x), KDE_REFERENCE_MAX, rng)
        bandwidths, log_k0 = [], []
        for kernel, metric in KDE_VARIANTS:
            grid = GridSearchCV(
                KernelDensity(kernel=kernel, metric=metric),
                {"bandwidth": list(BANDWIDTHS)},
                cv=min(GRID_CV, len(grid_rows)),
            ).fit(x[grid_rows])
            bandwidth = float(grid.best_params_["bandwidth"])
            single = KernelDensity(kernel=kernel, metric=metric, bandwidth=bandwidth)
            bandwidths.append(bandwidth)
            log_k0.append(single.fit(x[:1]).score_samples(x[:1])[0])
        return cls(
            bandwidths=np.array(bandwidths),
            log_k0=np.array(log_k0),
            reference_rows=reference_rows,
            reference_maccs=maccs[reference_rows].astype(np.uint8),
        )

    def log_density(self, maccs: np.ndarray, self_positions: np.ndarray) -> np.ndarray:
        """``(m, 3)`` exact log-densities, leave-one-out for reference molecules.

        The kernels are summed in log space (log-sum-exp) with scikit-learn's
        normalised kernels, ``log K(d) = log K(0) − d²/(2h²)`` (Gaussian) or
        ``log K(0) − d/h`` (exponential). This is exact where
        ``KernelDensity.score_samples`` approximates far-tail densities, and
        it avoids the catastrophic cancellation of subtracting ``K(0)`` from
        a total it dominates.

        Parameters
        ----------
        maccs : numpy.ndarray
            ``(m, 166)`` MACCS keys.
        self_positions : numpy.ndarray
            ``(m,)`` int: each row's position in the reference set, or ``-1``
            if it is not a reference molecule. Reference molecules have their
            own kernel removed.

        Returns
        -------
        numpy.ndarray
            ``(m, 3)`` log-densities, one column per ``KDE_NAMES``.
        """
        x = maccs.astype(np.float64)
        reference = self.reference_maccs.astype(np.float64)
        n_ref = len(reference)
        own = self_positions >= 0
        divisor = np.log(np.where(own & (n_ref > 1), n_ref - 1, n_ref))
        out = np.empty((len(x), len(KDE_VARIANTS)))
        binary = _is_binary(x) and _is_binary(reference)
        ref_counts = reference.sum(axis=1)
        for start in range(0, len(x), _CHUNK):
            stop = min(start + _CHUNK, len(x))
            rows = np.flatnonzero(own[start:stop])
            block = x[start:stop]
            if binary:
                # 0/1 keys: Manhattan distance = Hamming count = |a| + |b| − 2a·b
                # (one BLAS product, exact in float64), Euclidean = its root.
                hamming = block.sum(axis=1)[:, None] + ref_counts[None, :]
                hamming -= 2.0 * (block @ reference.T)
                np.maximum(hamming, 0.0, out=hamming)
                by_metric = {"manhattan": hamming, "euclidean": np.sqrt(hamming)}
            for j, (kernel, metric) in enumerate(KDE_VARIANTS):
                d = (
                    by_metric[metric]
                    if binary
                    else pairwise_distances(block, reference, metric=metric)
                )
                h = self.bandwidths[j]
                log_k = self.log_k0[j] - (
                    d * d / (2.0 * h * h) if kernel == "gaussian" else d / h
                )
                if n_ref > 1:
                    log_k[rows, self_positions[start:stop][rows]] = -np.inf
                out[start:stop, j] = logsumexp(log_k, axis=1)
        return out - divisor[:, None]


def _is_binary(values: np.ndarray) -> bool:
    """Whether every entry is 0 or 1 (MACCS keys)."""
    return bool(np.isin(values, (0.0, 1.0)).all())


def _subset(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """All rows when ``n <= cap``, else ``cap`` sorted random rows."""
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, size=cap, replace=False))
