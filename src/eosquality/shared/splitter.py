"""Deterministic 80/10/10 reference-library splitter.

Used by fit-time diagnostics that need a held-out validation / test
slice of the reference (e.g. the future Signal score). The same seed
and the same ratios are used for every fit, so consumers get a single
stable split per ``(library, n_ref)`` pair — no risk of two scores
disagreeing on what's "the held-out slice".

The split is just a shuffled partition: rows are permuted under a
fixed seed, then sliced into 80% / 10% / 10%. No stratification —
reference labels span many output columns with different shapes, and
the diagnostic users only need a random hold-out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEED = 0
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# TEST_FRAC is 1 - TRAIN_FRAC - VAL_FRAC = 0.1; kept implicit so the three
# always sum to 1 exactly even with future tweaks.


@dataclass(frozen=True)
class Split:
    """Three index arrays (row indices into the reference library)."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


class Splitter:
    """Fixed 80 / 10 / 10 shuffled split of the reference library.

    The seed and ratios are class constants — there is no constructor
    configuration. If a caller wants a different split, they should
    use a different splitter (not parameterise this one), so that
    "the canonical split for this library" is unambiguous.
    """

    SEED = SEED
    TRAIN_FRAC = TRAIN_FRAC
    VAL_FRAC = VAL_FRAC
    TEST_FRAC = 1.0 - TRAIN_FRAC - VAL_FRAC

    def split(self, n: int) -> Split:
        """Return shuffled train / val / test indices for ``n`` rows.

        ``n_train + n_val + n_test == n`` is guaranteed; any rounding
        residue is absorbed by the test slice.
        """
        if n <= 0:
            raise ValueError(f"Splitter requires n > 0; got n={n}.")
        rng = np.random.default_rng(self.SEED)
        perm = rng.permutation(n).astype(np.int64)
        n_train = int(round(n * self.TRAIN_FRAC))
        n_val = int(round(n * self.VAL_FRAC))
        return Split(
            train_indices=perm[:n_train],
            val_indices=perm[n_train : n_train + n_val],
            test_indices=perm[n_train + n_val :],
        )
