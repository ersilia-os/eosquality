"""Extremity score: how far each scaled value sits from the per-column center.

Extremity is position-based: per-feature extremity is the absolute value
of the eosframes-scaled value, clipped at 1.0. The most central value
(scaled to 0) scores 0; values at the rails (scaled to ±1 or beyond)
score 1. Complementary to typicality (which is density-based, not
position-based) — the pair (extremity, typicality) describes a query
better than either alone.

NaN queries carry no information about extremity, so they are dropped
from the aggregate (per-feature stays NaN). Aggregated across features
by arithmetic mean over non-NaN values.
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
from eosquality.scores._helpers import _component_metadata, _make_pipeline
from eosquality.shared.fit import fit_shared
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


SUBFOLDER = "extremity"
STATE_FILE = "state.json"
METADATA_FILE = "metadata.json"


@dataclass
class ExtremityRunResult:
    """Result returned by :meth:`Extremity.run`."""

    score: pd.Series  # (n_query,) aggregate extremity
    per_feature: pd.DataFrame  # (n_query, n_features)
    metadata: dict[str, Any] = field(default_factory=dict)


class Extremity:
    """Position-based per-feature extremity scorer.

    Holds one piece of fitted state:

    - ``reference_extremity_`` — mean aggregate extremity of the reference
      under its own scaler. A calibration anchor for downstream readers.

    Depends only on :class:`SharedFitState` — no vector index required, no
    per-column LUTs.
    """

    def __init__(self) -> None:
        self._shared: SharedFitState | None = None
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

        ref_scaled = _make_pipeline(shared).transform(reference)
        _, ref_agg = compute_extremity(scaled_values=ref_scaled)

        self._shared = shared
        self._reference_extremity = float(np.nanmean(ref_agg))
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

        validate_against_schema(query, self._shared.schema)

        if query_repr is None:
            query_repr = _make_pipeline(self._shared).transform(query)

        per_feature, score = compute_extremity(scaled_values=query_repr)
        per_feature_df = pd.DataFrame(
            per_feature,
            index=list(query.index),
            columns=list(self._shared.schema.column_names),
        )
        score_series = pd.Series(score, index=list(query.index), name="extremity")
        return ExtremityRunResult(
            score=score_series,
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

        Writes two files under ``extremity/``:

        - ``state.json`` — the ``reference_extremity`` baseline.
        - ``metadata.json`` — fit timestamp, fit duration, n_samples,
          n_features, eosquality_version.

        Also writes the shared subfolder via :func:`save_shared` so the
        artifact is self-contained.
        """
        self._check_fitted()
        assert self._shared is not None
        save_shared(self._shared, root)
        folder = pathlib.Path(root) / SUBFOLDER
        folder.mkdir(parents=True, exist_ok=True)
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
        """Reconstruct from ``<root>/shared/`` + ``<root>/extremity/state.json``."""
        shared = load_shared(root)
        folder = pathlib.Path(root) / SUBFOLDER
        with open(folder / STATE_FILE) as f:
            payload = json.load(f)
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
        instance._reference_extremity = float(payload["reference_extremity"])
        instance._fit_duration_seconds = fit_duration
        instance._fit_timestamp = fit_timestamp
        return instance

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        return self._shared is not None and self._reference_extremity is not None

    @property
    def shared_(self) -> SharedFitState:
        self._check_fitted()
        assert self._shared is not None
        return self._shared

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
        ``(n_query,)`` arithmetic mean of ``per_feature`` across features,
        ignoring NaN. A query whose every feature is NaN returns NaN.
    """
    n_query = scaled_values.shape[0]
    n_features = scaled_values.shape[1] if scaled_values.ndim > 1 else 0
    if n_features == 0:
        return np.zeros((n_query, 0)), np.zeros(n_query)

    per_feature = np.minimum(np.abs(scaled_values), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        aggregate = np.nanmean(per_feature, axis=1)
    return per_feature, aggregate
