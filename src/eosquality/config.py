from dataclasses import dataclass, field


@dataclass
class DistanceConfig:
    """Configuration for distance computation."""

    numeric_scaler: str = "type_aware"  # type-aware percentile normalization (only supported value)
    mixed_strategy: str = "weighted_mean"
    vector_metric: str = "cosine"


@dataclass
class NeighborConfig:
    """Configuration for nearest-neighbor search."""

    k: int = 20
    backend: str = "sklearn"  # "sklearn" | "faiss" (faiss requires optional dep)
    algorithm: str = "auto"   # passed to sklearn NearestNeighbors


@dataclass
class CoreConfig:
    """Configuration for core/reliable-subset detection (Phase 2)."""

    enabled: bool = False
    method: str = "hdbscan"
    min_cluster_size: int = 30


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap-based stability estimation (Phase 2)."""

    enabled: bool = False
    n_resamples: int = 50
    subsample_fraction: float = 0.8


@dataclass
class AggregationConfig:
    """Configuration for score aggregation."""

    strategy: str = "geometric_mean"  # how to combine component scores


@dataclass
class ErsiliaQualityConfig:
    """Top-level configuration for ErsiliaQuality."""

    distance: DistanceConfig = field(default_factory=DistanceConfig)
    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)

    @classmethod
    def default(cls) -> "ErsiliaQualityConfig":
        return cls()
