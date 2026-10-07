"""Median distance between two random reference-library molecules in physchem space.

    python scripts/physchem_pair_median.py [N_PAIRS] [SEED]   # default 1,000,000 pairs

Reproduces ``PAIR_MEDIAN`` in ``eosquality.scores._physchem_domain``: random
pairs of the canonical library's scaled physchem descriptors (``physchem_scaled.npy``),
clipped to ``+/-CLIP``, Euclidean distance, median. It is where the physchem
similarity ``1 - d / PAIR_MEDIAN`` is 0.
"""

import glob
import sys

import numpy as np

from eosquality.scores._physchem_domain import CLIP, PAIR_MEDIAN

CHUNK = 50_000


def main(n_pairs: int = 1_000_000, seed: int = 0) -> None:
    """Print the median random-pair distance and the value in the package.

    Parameters
    ----------
    n_pairs : int, optional
        Number of random pairs.
    seed : int, optional
        Seed of the pair sampling.
    """
    found = glob.glob("data/indices/*/physchem_scaled.npy")
    if not found:
        raise SystemExit("canonical library not found under data/indices/.")
    matrix = np.load(found[0], mmap_mode="r")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(matrix), n_pairs)
    j = rng.integers(0, len(matrix), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    distance = np.empty(len(i))
    for start in range(0, len(i), CHUNK):
        sel = slice(start, start + CHUNK)
        a = np.clip(matrix[np.sort(i[sel])], -CLIP, CLIP)  # sorted: faster mmap reads
        order = np.argsort(np.argsort(i[sel]))
        a = a[order].astype(np.float64)
        b = np.clip(matrix[np.sort(j[sel])], -CLIP, CLIP)
        b = b[np.argsort(np.argsort(j[sel]))].astype(np.float64)
        distance[sel] = np.sqrt(((a - b) ** 2).sum(axis=1))
    low, high = np.percentile(distance, [49.9, 50.1])
    print(
        f"{len(distance):,} pairs: median {np.median(distance):.3f} (49.9-50.1th percentile {low:.3f}-{high:.3f})"
    )
    print(f"PAIR_MEDIAN in the package: {PAIR_MEDIAN}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:3]]
    main(*args)
