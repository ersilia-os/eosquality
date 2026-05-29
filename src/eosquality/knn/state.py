"""KnnFitState: shared FP-kNN artifacts used by Support and Consistency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnnFitState:
    """Shared kNN artifacts for index-aware scores.

    One field is always present (and persisted):

    - ``k`` — number of neighbors used.

    The vector index itself is *not* persisted as a path. It is
    resolved at run time via the canonical library resolver in
    :mod:`eosquality.library.identity`, using
    ``shared.metadata.library_id`` as the key. This keeps saved
    artifacts portable across machines.

    The scaled reference matrix (``ref_repr``) used to live here too,
    but is now sourced from ``SharedFitState.ref_repr`` to avoid a
    duplicated 100MB-scale ``.npy`` on disk. Consistency reads
    ``shared.ref_repr`` directly when computing output-space neighbor
    distances.

    Two more fields are populated at fit time and dropped on save/load:

    - ``mean_fp_distances`` — ``(n_ref,)`` mean Tanimoto distance from
      each reference row to its k FP neighbors. Drives Support's CDF
      baseline.
    - ``reference_knn_indices`` — ``(n_ref, k)`` FP-selected neighbor
      indices. Consistency uses these to compute output-space distances
      against ``shared.ref_repr`` in its own fit.

    Both fit-only fields are ``None`` after save/load — re-fit if you
    need them.
    """

    k: int
    mean_fp_distances: np.ndarray | None = None
    reference_knn_indices: np.ndarray | None = None
