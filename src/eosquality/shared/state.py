"""SharedFitState: the fit artifact every score depends on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eosquality.shared.metadata import FitMetadata
from eosquality.schema.models import Schema


@dataclass
class SharedFitState:
    """Schema + eosframes scaler + binary class freqs + metadata + reference ids.

    Computed once per fit pass (by :func:`fit_shared`) and consumed by every
    score component. Persisted under ``<root>/shared/``.
    """

    schema: Schema
    scaler_params: dict[str, Any]
    binary_class_freq: dict[str, float]
    metadata: FitMetadata
    reference_ids: list[Any]
