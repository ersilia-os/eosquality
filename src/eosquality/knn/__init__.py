"""Shared kNN fit state for index-aware scores.

Support and Consistency both consume the same FP-selected neighbor indices
plus the output-space distances to those neighbors. ``fit_knn`` computes
this once; each score then derives its own per-score reduction (sorted
self-distances for Support, median k-distance for Consistency).
"""

from eosquality.knn.fit import fit_knn
from eosquality.knn.load import load_knn
from eosquality.knn.save import save_knn
from eosquality.knn.state import KnnFitState

__all__ = ["KnnFitState", "fit_knn", "save_knn", "load_knn"]
