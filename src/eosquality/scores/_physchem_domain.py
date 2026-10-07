"""Physicochemical applicability domain of one training set.

The distance-to-training-set domain, computed in physicochemical space
rather than fingerprint space: the mean Euclidean distance from a query to
its ``k`` nearest training molecules, over standardised RDKit physchem
descriptors.

This is deliberately the same method as :mod:`training_distance`, in a
different space. Distance to the training set is the oldest and still the
least-beaten applicability-domain signal:

    Sheridan, R. P.; Feuston, B. P.; Maiorov, V. N.; Kearsley, S. K.
    "Similarity to Molecules in the Training Set Is a Good Discriminator for
    Prediction Accuracy in QSAR." *J. Chem. Inf. Comput. Sci.* 2004, 44 (6),
    1912-1928.

Three implementation notes, each with a reason:

The standardised training matrix is kept with the artifact (float32, about
34 MB for a 39,000-molecule column) because a nearest-neighbour domain needs
the reference molecules themselves, not a summary of them.

- **Standardise first, in a fixed space.** Euclidean distance over raw
  descriptors is dominated by whichever has the largest scale (molecular
  weight swamps everything). Each descriptor is centred and scaled with the
  *reference library's* scaler (median imputation, then mean and standard
  deviation over the 1.35M-molecule library, shipped with the package as
  ``library/physchem_scaler.json``) rather than the training set's own
  statistics. Every model then shares one descriptor space, so a raw distance
  means the same thing everywhere: "library standard deviations". Scaled
  values are clipped to ``+/-CLIP`` because the library scaler is not robust
  to outliers: ``Ipc`` grows exponentially with molecule size (library mean
  4e8, scale 5e9) and reached a z-score of 2.7e26 on eos4e40, and a rare
  fragment count such as ``fr_isothiocyan`` (scale 0.02) gives z ~ 100 for a
  single occurrence. Non-finite cells are replaced by the library median first,
  since RDKit returns NaN for a descriptor that overflows or fails.
- **Leave one out.** The calibration table is built from each training
  molecule's distance to its ``k`` nearest *other* training molecules.
  Including itself would make every training molecule's first neighbour an
  exact zero and halve its apparent distance, so the calibration would say
  queries are far more novel than they are. (This matters here, unlike for
  an ellipsoid-style measure where one molecule barely moves a centroid.)
- **No cutoff.** A widely used threshold is ``D = d̄ + Z·σ`` over the
  training distances with ``Z = 0.5`` (Golbraikh et al., *J. Comput. Aided
  Mol. Des.* 2003, 17, 241-253). It is not an OECD prescription, despite
  common belief - the guidance reviews ranges, convex hull, centroid
  distance, leverage and density methods, and endorses none of them. We
  report the mid-rank percentile instead, which carries the same information
  without assuming the distances are normal; a user who wants the convention
  can recover it, since on eos4e40 ``Z = 0.5`` falls at the 65th percentile.

An ellipsoid-style alternative (Hotelling T², Mahalanobis, leverage) was
implemented first and removed. Leverage is an ordinary-least-squares
construct - "The original leverage can only be applied in the case of a
linear regression, but not for non-linear regression" (Dutschmann, Schlenker
& Baumann, *Mol. Inf.* 2024, 43, e202400018) - and in the one published
head-to-head benchmark a thresholded Mahalanobis domain made external Q²
*worse* than using no domain at all (0.797 -> 0.791) while discarding 6 of 95
test compounds, on a radial-basis neural network rather than a linear model
(Sahigara et al., *J. Cheminform.* 2013, 5, 27). T² is also a latent-variable
construct, and its companion DModX is the PLS X-residual, which a random
forest does not have.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

# Neighbours averaged per molecule, matching K_NEIGHBORS in training_distance.
K_NEIGHBORS = 5
# Median Euclidean distance between two random reference-library molecules in this
# scaled, clipped space: 1,000,000 random pairs, seed 0, canonical library v0
# (18.701; seed 7 gives 18.700)
# (``scripts/physchem_pair_median.py`` recomputes it). It anchors the similarity.
PAIR_MEDIAN = 18.70
# Scaled descriptor values are clipped to [-CLIP, CLIP] (see the module docstring).
CLIP = 10.0
# A neighbour at or below this distance is the molecule itself: standardisation
# is deterministic, so a training molecule reproduces its stored row exactly.
_SELF_MATCH_DISTANCE = 1e-9
STATE_FILE = "physchem_domain.json"
ARRAYS_FILE = "physchem_domain.npz"


@dataclass
class PhyschemDomain:
    """Nearest-neighbour physchem domain of one training set."""

    median: np.ndarray  # (p,) library median per descriptor, for non-finite cells
    mean: np.ndarray  # (p,) library mean (standardisation centre)
    scale: np.ndarray  # (p,) library scale (1 where constant)
    train: np.ndarray  # (n, p) scaled, clipped training molecules, float32
    k: int  # neighbours averaged
    clip: float  # scaled values are clipped to [-clip, clip]
    pair_median: float  # distance at which the similarity is 0
    sorted_distances: np.ndarray  # ascending leave-one-out training distances

    @property
    def n_train(self) -> int:
        """Number of training molecules the domain is built on.

        Returns
        -------
        int
        """
        return len(self.train)

    @classmethod
    def fit(
        cls,
        raw: np.ndarray,
        scaler: dict,
        k: int = K_NEIGHBORS,
        clip: float = CLIP,
        pair_median: float = PAIR_MEDIAN,
    ) -> PhyschemDomain:
        """Fit the domain on a training set's raw physchem matrix.

        Parameters
        ----------
        raw : numpy.ndarray
            ``(n, p)`` descriptors, non-finite cells allowed.
        scaler : dict
            Library scaler parameters (``median``, ``mean``, ``scale``), as
            from :func:`eosquality.library.physchem.canonical_scaler`.
        k : int, optional
            Neighbours to average; capped at ``n - 1``.
        clip : float, optional
            Scaled values are clipped to ``[-clip, clip]``.
        pair_median : float, optional
            Distance between two random library molecules, where the similarity
            is 0.

        Returns
        -------
        PhyschemDomain

        Raises
        ------
        ValueError
            If the scaler does not have one entry per descriptor of ``raw``.
        """
        x = np.asarray(raw, dtype=np.float64)
        median = np.asarray(scaler["median"], dtype=np.float64)
        mean = np.asarray(scaler["mean"], dtype=np.float64)
        scale = np.asarray(scaler["scale"], dtype=np.float64)
        if not len(median) == len(mean) == len(scale) == x.shape[1]:
            raise ValueError(
                f"scaler has {len(median)} descriptors, the matrix {x.shape[1]}."
            )
        scale = np.where(scale > 0, scale, 1.0)
        train = _standardise(_fill(x, median), mean, scale, clip)
        k = max(1, min(k, len(train) - 1))
        domain = cls(
            median=median,
            mean=mean,
            scale=scale,
            train=train,
            k=k,
            clip=float(clip),
            pair_median=float(pair_median),
            sorted_distances=np.zeros(0),
        )
        # Leave-one-out: each training molecule against its k nearest *other*
        # training molecules, so the calibration is what a query would see.
        loo = domain._mean_distance(train)
        domain.sorted_distances = np.sort(loo[np.isfinite(loo)])
        return domain

    def similarity(self, distance: np.ndarray) -> np.ndarray:
        """Similarity from a distance: ``1 - d / pair_median``.

        1 is identical and 0 is no closer than two random reference-library
        molecules. It is deliberately not clipped, so it goes negative for a
        molecule further away than a random pair and keeps its ordering.

        Parameters
        ----------
        distance : numpy.ndarray
            Distances from :meth:`measure`.

        Returns
        -------
        numpy.ndarray
        """
        return 1.0 - np.asarray(distance, dtype=np.float64) / self.pair_median

    def measure(self, raw: np.ndarray) -> np.ndarray:
        """Mean distance to the ``k`` nearest training molecules.

        Parameters
        ----------
        raw : numpy.ndarray
            ``(n, p)`` raw descriptors, non-finite cells allowed.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` mean Euclidean distance in standardised space.
        """
        x = np.asarray(raw, dtype=np.float64)
        if not len(x):
            return np.zeros(0)
        return self._mean_distance(
            _standardise(_fill(x, self.median), self.mean, self.scale, self.clip)
        )

    def _mean_distance(self, z: np.ndarray) -> np.ndarray:
        """Mean distance to the k nearest training rows, excluding any self match.

        ``k + 1`` neighbours are fetched and the zero-distance one dropped when
        present, so a molecule that is itself in the training set is scored
        against its k nearest *others* - the same rule
        ``_helpers._query_fp_distances`` applies on the fingerprint side. Rows
        without a self match drop their furthest neighbour instead, so every
        row averages exactly k.
        """
        from sklearn.neighbors import NearestNeighbors

        wanted = min(self.k + 1, self.n_train)
        nn = NearestNeighbors(n_neighbors=wanted).fit(self.train)
        distances, _ = nn.kneighbors(z)
        if wanted == 1:
            return distances[:, 0]
        is_self = distances[:, 0] <= _SELF_MATCH_DISTANCE
        kept = np.where(is_self[:, None], distances[:, 1:], distances[:, :-1])
        return kept.mean(axis=1)

    def save(self, folder: pathlib.Path) -> None:
        """Write the arrays and a small JSON header into ``folder``.

        Parameters
        ----------
        folder : pathlib.Path
            Destination (created if missing).
        """
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / ARRAYS_FILE,
            median=self.median,
            mean=self.mean,
            scale=self.scale,
            train=self.train,
            sorted_distances=self.sorted_distances,
        )
        with open(folder / STATE_FILE, "w") as f:
            json.dump(
                {
                    "k": self.k,
                    "n_train": self.n_train,
                    "clip": self.clip,
                    "pair_median": self.pair_median,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, folder: pathlib.Path) -> PhyschemDomain:
        """Read a domain written by :meth:`save`.

        Parameters
        ----------
        folder : pathlib.Path
            Folder written by :meth:`save`.

        Returns
        -------
        PhyschemDomain
        """
        with open(folder / STATE_FILE) as f:
            state = json.load(f)
        with np.load(folder / ARRAYS_FILE) as arrays:
            return cls(
                median=arrays["median"],
                mean=arrays["mean"],
                scale=arrays["scale"],
                train=arrays["train"],
                k=int(state["k"]),
                clip=float(state["clip"]),
                pair_median=float(state["pair_median"]),
                sorted_distances=arrays["sorted_distances"],
            )


def _standardise(
    filled: np.ndarray, mean: np.ndarray, scale: np.ndarray, clip: float
) -> np.ndarray:
    """Centre, scale and clip; float32 so a training molecule reproduces its row."""
    return np.clip((filled - mean) / scale, -clip, clip).astype(np.float32)


def _fill(x: np.ndarray, median: np.ndarray) -> np.ndarray:
    """``x`` with every non-finite cell replaced by its column median."""
    return np.where(np.isfinite(x), x, median)
