"""KnnFitState: shared FP-kNN artifacts used by Support and Consistency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KnnFitState:
    """Shared kNN artifacts for index-aware scores.

    One field is always present (and persisted):

    - ``k`` — number of neighbors used.

    The vector index is not part of this state: at run time it is
    resolved from ``shared.metadata`` (canonical library by
    ``library_id``, or a recorded custom index path). The scaled
    reference matrix lives in ``SharedFitState.ref_repr``.

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
