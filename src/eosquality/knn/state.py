"""KnnFitState: shared FP-kNN artifacts used by Support and Consistency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnnFitState:
    """Shared kNN artifacts for index-aware scores.

    Two fields are always present (and persisted):

    - ``ref_repr`` — scaled reference array ``(n_ref, n_features)``.
    - ``k`` — number of neighbors used.

    The vector index itself is *not* persisted as a path. It is
    resolved at run time via the canonical library resolver in
    :mod:`eosquality.library.identity`, using
    ``shared.metadata.library_id`` as the key. This keeps saved
    artifacts portable across machines.

    Two more fields are populated at fit time and dropped on save/load:

    - ``mean_fp_distances`` — ``(n_ref,)`` mean Tanimoto distance from
      each reference row to its k FP neighbors. Drives Support's CDF
      baseline.
    - ``reference_knn_indices`` — ``(n_ref, k)`` FP-selected neighbor
      indices. Consistency uses these to compute output-space distances
      against ``ref_repr`` in its own fit.

    Both fit-only fields are ``None`` after save/load — re-fit if you
    need them.
    """

    ref_repr: np.ndarray
    k: int
    mean_fp_distances: np.ndarray | None = None
    reference_knn_indices: np.ndarray | None = None
