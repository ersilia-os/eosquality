"""SharedFitState: the fit artifact every score depends on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eosquality.schema.models import Schema
from eosquality.shared.metadata import FitMetadata


@dataclass
class SharedFitState:
    """Schema + eosframes scaler + binary class freqs + metadata + reference ids
    + selected feature subset.

    Computed once per fit pass (by :func:`fit_shared`) and consumed by every
    score component. Persisted under ``<root>/shared/``.

    The ``selected_columns`` field carries the post-reduction subset of
    feature column names. It is always a subset (or full copy) of
    ``schema.column_names``, preserving original order. When no reduction
    is applied it equals the full column list, so downstream consumers can
    treat it uniformly via :meth:`filter_features`.

    The ``ref_repr`` field carries the eosframes-scaled reference matrix
    projected onto ``selected_columns`` — shape ``(n_ref, n_selected)``.
    Set by :func:`fit_shared` after the scaler runs; score fits reuse it
    instead of re-scaling the reference. It is not saved (``None`` after
    loading): no score reads it at run time.
    """

    schema: Schema
    scaler_params: dict[str, Any]
    binary_class_freq: dict[str, float]
    metadata: FitMetadata
    reference_ids: list[Any]
    selected_columns: list[str] = field(default_factory=list)
    ref_repr: np.ndarray | None = None

    def selected_indices(self) -> np.ndarray:
        """Map ``selected_columns`` back to positional indices in ``schema``.

        Returns
        -------
        numpy.ndarray
        """
        name_to_idx = {n: i for i, n in enumerate(self.schema.column_names)}
        return np.asarray(
            [name_to_idx[c] for c in self.selected_columns], dtype=np.int64
        )

    def filter_features(self, arr: np.ndarray) -> np.ndarray:
        """Project ``arr`` (``..., n_full_features``) onto the selected columns.

        Returns ``arr`` unchanged when no reduction was applied (the
        selected set equals the full schema). Otherwise slices the trailing
        axis to the selected indices.

        Parameters
        ----------
        arr : numpy.ndarray
            Array whose last axis has one entry per schema column.

        Returns
        -------
        numpy.ndarray
        """
        if len(self.selected_columns) == len(self.schema.column_names):
            return arr
        return arr[..., self.selected_indices()]
