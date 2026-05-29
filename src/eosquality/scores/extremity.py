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
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import (
    AGGREGATE_QUANTILE,
    _component_metadata,
    _make_query_repr,
    _score_from_aggregates,
)
from eosquality.shared.fit import fit_shared
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


SUBFOLDER = "extremity"
STATE_FILE = "state.json"
METADATA_FILE = "metadata.json"
SELF_AGGREGATES_FILE = "reference_self_aggregates.npy"


@dataclass
class ExtremityRunResult:
    """Result returned by :meth:`Extremity.run`."""

    score: pd.Series  # (n_query,) calibrated aggregate extremity in [0, 1]
    score_raw: pd.Series  # (n_query,) Q66 aggregate before CDF lookup, in [0, 1]
    per_feature: pd.DataFrame  # (n_query, n_features)
    metadata: dict[str, Any] = field(default_factory=dict)


class Extremity:
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

    def __init__(self) -> None:
        self._shared: SharedFitState | None = None
        self._sorted_self_aggregates: np.ndarray | None = None  # (n_ref,)
        self._reference_extremity: float | None = None
        self._fit_duration_seconds: float | None = None
        self._fit_timestamp: str | None = None

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

        Scales the reference with the shared eosframes pipeline and records
        ``reference_extremity_`` — the mean aggregate extremity of the
        reference under its own scaler — as a calibration baseline.

        Either pass a pre-fit ``shared=`` (when composed by ErsiliaQuality),
        or pass ``eos_id`` + ``version`` so Extremity can fit the shared
        state itself.

        Records the wall-clock duration and a UTC timestamp; both are
        persisted by :meth:`save` to ``extremity/metadata.json``.
        """
        t0 = time.perf_counter()
        if shared is None:
            if eos_id is None or version is None:
                raise ValueError(
                    "Extremity.fit needs either a pre-fit shared= argument, "
                    "or eos_id= and version= so it can fit the shared state itself."
                )
            shared, _ref_repr = fit_shared(reference, eos_id=eos_id, version=version)
        else:
            validate_against_schema(reference, shared.schema)

        ref_scaled = _make_query_repr(shared, reference)
        _, ref_agg = compute_extremity(scaled_values=ref_scaled)

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
        reference_extremity = float(
            np.nanmean(
                _score_from_aggregates(
                    ref_agg, sorted_self_aggregates, sorted_self_aggregates.size
                )
            )
        )

        self._shared = shared
        self._sorted_self_aggregates = sorted_self_aggregates
        self._reference_extremity = reference_extremity
        self._fit_duration_seconds = float(time.perf_counter() - t0)
        self._fit_timestamp = datetime.now(tz=timezone.utc).isoformat()
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
            Optional pre-scaled query array (output of the eosframes scaler)
            ``(n_query, n_features)``. If provided, the eosframes transform
            step is skipped — used by ErsiliaQuality to share scaling work
            across multiple scores.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._sorted_self_aggregates is not None

        validate_against_schema(query, self._shared.schema)

        if query_repr is None:
            query_repr = _make_query_repr(self._shared, query)

        per_feature, raw_aggregate = compute_extremity(scaled_values=query_repr)
        n_ref = self._sorted_self_aggregates.size
        score = _score_from_aggregates(
            raw_aggregate, self._sorted_self_aggregates, n_ref
        )
        # Preserve the NaN-row semantics: a query whose every feature is
        # NaN has no information about extremity, so the calibrated score
        # follows the raw NaN aggregate.
        nan_rows = ~np.isfinite(raw_aggregate)
        if nan_rows.any():
            score = np.where(nan_rows, np.nan, score)
        per_feature_df = pd.DataFrame(
            per_feature,
            index=list(query.index),
            columns=list(self._shared.selected_columns),
        )
        score_series = pd.Series(score, index=list(query.index), name="extremity")
        score_raw_series = pd.Series(
            raw_aggregate, index=list(query.index), name="extremity_raw"
        )
        return ExtremityRunResult(
            score=score_series,
            score_raw=score_raw_series,
            per_feature=per_feature_df,
            metadata={
                "reference_extremity": self._reference_extremity,
                "n_reference": len(self._shared.reference_ids),
            },
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, root: str | pathlib.Path) -> pathlib.Path:
        """Persist into ``<root>/shared/`` and ``<root>/extremity/``.

        Writes three files under ``extremity/``:

        - ``state.json`` — the ``reference_extremity`` baseline.
        - ``reference_self_aggregates.npy`` — sorted reference per-row
          Q66 aggregates; the CDF lookup table.
        - ``metadata.json`` — fit timestamp, fit duration.

        Also writes the shared subfolder via :func:`save_shared` so the
        artifact is self-contained.
        """
        self._check_fitted()
        assert self._shared is not None
        assert self._sorted_self_aggregates is not None
        save_shared(self._shared, root)
        folder = pathlib.Path(root) / SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / SELF_AGGREGATES_FILE, self._sorted_self_aggregates)
        payload = {"reference_extremity": self._reference_extremity}
        with open(folder / STATE_FILE, "w") as f:
            json.dump(payload, f)
        meta = _component_metadata(
            component="extremity",
            k=None,
            fit_timestamp=self._fit_timestamp,
            fit_duration_seconds=self._fit_duration_seconds,
        )
        with open(folder / METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug(
            f"  extremity/ | reference_extremity={self._reference_extremity:.4f}"
            f" | fit_duration={meta['fit_duration_seconds']:.3f}s"
        )
        return pathlib.Path(root)

    @classmethod
    def load(cls, root: str | pathlib.Path) -> "Extremity":
        """Reconstruct from ``<root>/shared/`` + ``<root>/extremity/``."""
        shared = load_shared(root)
        folder = pathlib.Path(root) / SUBFOLDER
        with open(folder / STATE_FILE) as f:
            payload = json.load(f)

        aggregates_path = folder / SELF_AGGREGATES_FILE
        if not aggregates_path.is_file():
            raise FileNotFoundError(
                f"Missing {aggregates_path}. This artifact predates the "
                "Q66 + CDF-calibrated extremity format and must be refit "
                "with the current eosquality version."
            )
        sorted_self_aggregates = np.load(aggregates_path)

        meta_path = folder / METADATA_FILE
        fit_duration = None
        fit_timestamp = None
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            fit_duration = float(meta.get("fit_duration_seconds", 0.0))
            fit_timestamp = meta.get("fit_timestamp")

        instance = cls()
        instance._shared = shared
        instance._sorted_self_aggregates = sorted_self_aggregates
        instance._reference_extremity = float(payload["reference_extremity"])
        instance._fit_duration_seconds = fit_duration
        instance._fit_timestamp = fit_timestamp
        return instance

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return (
            self._shared is not None
            and self._sorted_self_aggregates is not None
            and self._reference_extremity is not None
        )

    @property
    def shared_(self) -> SharedFitState:
        self._check_fitted()
        assert self._shared is not None
        return self._shared

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

    @property
    def fit_duration_seconds_(self) -> float | None:
        return self._fit_duration_seconds

    @property
    def fit_timestamp_(self) -> str | None:
        return self._fit_timestamp

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Extremity must be fitted (or loaded) before use.")


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
