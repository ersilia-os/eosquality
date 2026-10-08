"""Median distance between two random reference-library molecules in physchem space.

    python scripts/physchem_pair_median.py [N_MOLECULES] [N_PAIRS] [SEED]
    # defaults: 20,000 molecules, 1,000,000 pairs

Reproduces ``PAIR_MEDIAN`` in ``eosquality.scores._physchem_domain``: a random
sample of the canonical library's molecules, their RDKit descriptors scaled with
the shipped library scaler and clipped to ``+/-CLIP``, then the Euclidean
distance of random pairs of the sample, and its median. It is where the
physchem similarity ``1 - d / PAIR_MEDIAN`` is 0.
"""

import sys

import numpy as np

from eosquality.library.physchem import canonical_scaler, compute_physchem_raw
from eosquality.library.reference import ReferenceLibrary
from eosquality.scores._physchem_domain import CLIP, PAIR_MEDIAN, _fill, _standardise


def main(n_molecules: int = 20_000, n_pairs: int = 1_000_000, seed: int = 0) -> None:
    """Print the median random-pair distance and the value in the package.

    Parameters
    ----------
    n_molecules : int, optional
        Library molecules sampled (their pairs are drawn).
    n_pairs : int, optional
        Number of random pairs.
    seed : int, optional
        Seed of the molecule and pair sampling.
    """
    smiles = ReferenceLibrary.load().smiles
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(smiles), size=min(n_molecules, len(smiles)), replace=False)
    raw = compute_physchem_raw([smiles[i] for i in picked])
    scaler = canonical_scaler()
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    matrix = _standardise(
        _fill(raw.astype(np.float64), np.asarray(scaler["median"])),
        np.asarray(scaler["mean"]),
        np.where(scale > 0, scale, 1.0),
        CLIP,
    )
    i = rng.integers(0, len(matrix), n_pairs)
    j = rng.integers(0, len(matrix), n_pairs)
    keep = i != j
    distance = np.sqrt(((matrix[i[keep]] - matrix[j[keep]]) ** 2).sum(axis=1))
    low, high = np.percentile(distance, [49.9, 50.1])
    print(
        f"{len(distance):,} pairs: median {np.median(distance):.3f} (49.9-50.1th percentile {low:.3f}-{high:.3f})"
    )
    print(f"PAIR_MEDIAN in the package: {PAIR_MEDIAN}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:4]]
    main(*args)
