"""FitState: the persisted artifact produced by ErsiliaQuality.fit()."""

from dataclasses import dataclass, field
from typing import Any

from eosquality.config import ErsiliaQualityConfig
from eosquality.reference.metadata import FitMetadata
from eosquality.schema.models import Schema


@dataclass
class ReferenceReport:
    """Summary statistics describing the quality of the reference population."""

    reference_quality: float
    cohesion_score: float
    fragmentation_score: float
    median_k_distance: float
    notes: list[str] = field(default_factory=list)


@dataclass
class FitState:
    """All artifacts produced during fit(), used by run() and serialization."""

    config: ErsiliaQualityConfig
    schema: Schema
    preprocess_state: dict[str, Any]
    reference_ids: list[Any]
    reference_repr: Any              # np.ndarray of scaled reference features
    reference_knn_distances: Any     # (n_ref, k) array of self-kNN distances (output-space)
    reference_knn_indices: Any       # (n_ref, k) array of self-kNN indices (FP-selected)
    reference_report: ReferenceReport
    metadata: FitMetadata
    vector_index_path: str           # absolute path to the VectorIndex folder
