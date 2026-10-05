"""Configuration dataclasses for :class:`~eosquality.quality.ErsiliaQuality`."""

from dataclasses import dataclass, field


@dataclass
class NeighborConfig:
    """Fingerprint nearest-neighbor settings."""

    k: int = 5  # neighbors per molecule; must be <= the index's max_k

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k!r}.")


@dataclass
class ErsiliaQualityConfig:
    """Top-level configuration for ErsiliaQuality."""

    neighbors: NeighborConfig = field(default_factory=NeighborConfig)
