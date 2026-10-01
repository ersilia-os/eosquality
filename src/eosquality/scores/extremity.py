"""Extremity score: how far each scaled value sits from the per-column center.

Extremity is position-based: per-feature extremity is the absolute value
of the eosframes-scaled value, clipped at 1.0. The most central value
(scaled to 0) scores 0; values at the rails (scaled to ±1 or beyond)
score 1. Complementary to typicality (which is density-based, not
position-based) — the pair (extremity, typicality) describes a query
better than either alone.

NaN queries carry no information about extremity, so they are dropped
from the aggregate (per-feature stays NaN). The per-row aggregate is the
**66th percentile** of per-feature values (``AGGREGATE_QUANTILE``), then
mapped through the reference's own sorted Q66 distribution via
:func:`_score_from_aggregates`. The calibrated score is uniform under
the reference: a query at the reference median scores ~0.5, and the
score is comparable across models with different feature counts.
"""

from __future__ import annotations

import json
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eosquality.schema.infer import validate_against_schema
from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import (
    AGGREGATE_QUANTILE,
    _make_query_repr,
    _reference_repr,
    _resolve_shared,
    _score_from_aggregates,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger

SUBFOLDER = "extremity"
STATE_FILE = "state.json"
SELF_AGGREGATES_FILE = "reference_self_aggregates.npy"


@dataclass
class ExtremityRunResult:
    """Result returned by :meth:`Extremity.run`."""

    score: pd.Series  # (n_query,) calibrated aggregate extremity in [0, 1]
    score_raw: pd.Series  # (n_query,) Q66 aggregate before CDF lookup, in [0, 1]
    per_feature: pd.DataFrame  # (n_query, n_features)
    metadata: dict[str, Any] = field(default_factory=dict)


class Extremity(ScoreComponent):
    """Position-based per-feature extremity scorer.

    Holds two pieces of fitted state:

    - ``sorted_self_aggregates_`` — ``(n_ref,)`` ascending array of
      reference per-row Q66 aggregates. The CDF lookup table that maps
      raw aggregates to calibrated scores.
    - ``reference_extremity_`` — mean reference-as-query calibrated
      extremity. ≈ 0.5 by construction; a sanity-check anchor.

    Depends only on :class:`SharedFitState` — no vector index required, no
    per-column LUTs.
    """

    NAME = SUBFOLDER

    def __init__(self) -> None:
        super().__init__()
        self._sorted_self_aggregates: np.ndarray | None = None  # (n_ref,)
        self._reference_extremity: float | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        *,
        eos_id: str | None = None,
        version: str | None = None,
        shared: SharedFitState | None = None,
    ) -> "Extremity":
        """Fit on a reference DataFrame.

        Builds the sorted reference Q66 aggregates used as the calibration
        CDF and records ``reference_extremity_`` as a sanity anchor.

        Either pass a pre-fit ``shared=`` (when composed by ErsiliaQuality),
        or pass ``eos_id`` + ``version`` so Extremity can fit the shared
        state itself.
        """
        t0 = time.perf_counter()
        shared = _resolve_shared(
            reference,
            shared=shared,
            eos_id=eos_id,
            version=version,
            component="Extremity",
        )
        _, ref_agg = compute_extremity(scaled_values=_reference_repr(shared, reference))

        # CDF calibration is built only over reference rows whose aggregate
        # is finite. An all-NaN reference row carries no information and
        # would corrupt np.sort + searchsorted with NaNs at the tail.
        finite_ref_agg = ref_agg[np.isfinite(ref_agg)]
        sorted_self_aggregates = np.sort(finite_ref_agg).astype(np.float64)
        if sorted_self_aggregates.size == 0:
            raise ValueError(
                "Extremity.fit: every reference row has all-NaN extremity, "
                "so no CDF calibration is possible. Check the reference data."
            )

        self._shared = shared
        self._sorted_self_aggregates = sorted_self_aggregates
        self._reference_extremity = float(
            np.nanmean(
                _score_from_aggregates(
                    ref_agg, sorted_self_aggregates, sorted_self_aggregates.size
                )
            )
        )
        self._finish_fit(t0)
        logger.debug(
            f"Extremity fit | reference_extremity={self._reference_extremity:.4f}"
            f" | duration={self._fit_duration_seconds:.3f}s"
        )
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        query: pd.DataFrame,
        *,
        query_repr: np.ndarray | None = None,
    ) -> ExtremityRunResult:
        """Score query samples.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference.
        query_repr:
            Optional pre-scaled, feature-selected query array
            ``(n_query, n_selected)``. If provided, schema validation and the
            eosframes transform are skipped — used by ErsiliaQuality to share
            that work across scores.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._sorted_self_aggregates is not None

        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)

        per_feature, raw_aggregate = compute_extremity(scaled_values=query_repr)
        score = _score_from_aggregates(
            raw_aggregate,
            self._sorted_self_aggregates,
            self._sorted_self_aggregates.size,
        )
        # A query whose every feature is NaN has no information about
        # extremity, so the calibrated score follows the raw NaN aggregate.
        score = np.where(np.isfinite(raw_aggregate), score, np.nan)
        idx = list(query.index)
        return ExtremityRunResult(
            score=pd.Series(score, index=idx, name="extremity"),
            score_raw=pd.Series(raw_aggregate, index=idx, name="extremity_raw"),
            per_feature=pd.DataFrame(
                per_feature, index=idx, columns=list(self._shared.selected_columns)
            ),
            metadata={
                "reference_extremity": self._reference_extremity,
                "n_reference": len(self._shared.reference_ids),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        """Write ``state.json`` (baseline) and the sorted CDF array."""
        assert self._sorted_self_aggregates is not None
        np.save(folder / SELF_AGGREGATES_FILE, self._sorted_self_aggregates)
        with open(folder / STATE_FILE, "w") as f:
            json.dump({"reference_extremity": self._reference_extremity}, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        payload = read_json(folder / STATE_FILE, self.NAME)
        self._sorted_self_aggregates = np.load(
            require_file(folder / SELF_AGGREGATES_FILE, self.NAME)
        )
        self._reference_extremity = float(payload["reference_extremity"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._sorted_self_aggregates is not None
            and self._reference_extremity is not None
        )

    @property
    def sorted_self_aggregates_(self) -> np.ndarray:
        self._check_fitted()
        assert self._sorted_self_aggregates is not None
        return self._sorted_self_aggregates

    @property
    def reference_extremity_(self) -> float:
        self._check_fitted()
        assert self._reference_extremity is not None
        return self._reference_extremity


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_extremity(
    scaled_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature and aggregate extremity from eosframes-scaled values.

    Parameters
    ----------
    scaled_values:
        ``(n_query, n_features)`` float array produced by the eosframes
        scaler (the output of :meth:`PreprocessPipeline.transform`).

    Returns
    -------
    per_feature:
        ``(n_query, n_features)`` extremity in ``[0, 1]``. NaN inputs stay
        NaN.
    aggregate:
        ``(n_query,)`` 66th-percentile of ``per_feature`` across features
        (see ``AGGREGATE_QUANTILE``), ignoring NaN. The shift away from
        the mean prevents the aggregate from collapsing to the
        per-feature expectation as ``n_features`` grows; downstream CDF
        calibration in :meth:`Extremity.run` further re-spreads it to
        uniform under the reference. A query whose every feature is NaN
        returns NaN.
    """
    n_query = scaled_values.shape[0]
    n_features = scaled_values.shape[1] if scaled_values.ndim > 1 else 0
    if n_features == 0:
        return np.zeros((n_query, 0)), np.zeros(n_query)

    per_feature = np.minimum(np.abs(scaled_values), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        aggregate = np.nanquantile(per_feature, AGGREGATE_QUANTILE, axis=1)
    return per_feature, aggregate
