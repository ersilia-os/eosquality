"""Configuration dataclasses for :class:`~eosquality.quality.ErsiliaQuality`."""

from dataclasses import dataclass, field


@dataclass
class NeighborConfig:
    """Fingerprint nearest-neighbor settings."""

    k: int = 5  # neighbors per molecule; must be <= the index's max_k


@dataclass
class ErsiliaQualityConfig:
    """Top-level configuration for ErsiliaQuality."""

    neighbors: NeighborConfig = field(default_factory=NeighborConfig)

    @classmethod
    def default(cls) -> "ErsiliaQualityConfig":
        return cls()
