"""Shared kNN fit state for Support and Consistency.

``fit_knn`` slices the reference's precomputed FP self-kNN (indices +
Tanimoto distances) from the vector index once per fit. Support reduces the
distances to its CDF; Consistency uses the indices to compute output-space
neighbor distances against ``shared.ref_repr``. Only ``k`` is persisted.
"""

from eosquality.knn.fit import fit_knn
from eosquality.knn.load import load_knn
from eosquality.knn.save import save_knn
from eosquality.knn.state import KnnFitState

__all__ = ["KnnFitState", "fit_knn", "save_knn", "load_knn"]
