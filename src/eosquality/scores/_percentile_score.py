"""Shared machinery of the two output-based scores, typicality and extremity.

Both turn the scaled value of every selected output column into a per-feature
value (:meth:`PercentileScore._per_feature`), place it on that column's own
reference distribution (a per-column percentile, higher = more of the score),
combine the percentiles across columns at the 66th percentile, and map that
aggregate through the reference's own distribution of it, so the reference
scores ~Uniform(0, 1) whatever the number of columns. The raw column is the
66th percentile of the per-feature values themselves. Subclasses define the
per-feature value, how a column's reference distribution is stored, and its
files.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from eosquality.schema.infer import validate_against_schema
from eosquality.scores._base import ScoreComponent, read_json, require_file
from eosquality.scores._helpers import (
    _aggregate_percentiles,
    _cdf_score,
    _make_query_repr,
    _nan_aggregate,
    _reference_repr,
    _resolve_shared,
    _sorted_finite,
)
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger

STATE_FILE = "state.json"
SELF_AGGREGATES_FILE = "reference_self_aggregates.npy"


@dataclass
class PercentileRunResult:
    """Result returned by ``Typicality.run`` and ``Extremity.run``."""

    score: pd.Series  # (n_query,) whole-model percentile, in (0, 1]
    score_raw: pd.Series  # (n_query,) Q66 of the per-feature values
    per_feature: pd.DataFrame  # (n_query, n_features) per-feature values
    per_feature_pct: pd.DataFrame  # (n_query, n_features) per-column percentiles
    metadata: dict[str, Any] = field(default_factory=dict)


class PercentileScore(ScoreComponent):
    """Base of :class:`Typicality` and :class:`Extremity` (shared state only)."""

    #: Public score name, e.g. ``"ref_typicality"``; the Series are ``<SCORE>_pct``
    #: and ``<SCORE>_raw``.
    SCORE: ClassVar[str] = ""

    def __init__(self) -> None:
        super().__init__()
        self._sorted_self_aggregates: np.ndarray | None = None  # (n_ref,)
        self._anchor: float | None = None

    # -- what a subclass provides -------------------------------------------

    def _per_feature(self, scaled: np.ndarray) -> np.ndarray:
        """Per-feature values ``(n, n_features)`` of scaled values; NaN stays NaN."""
        raise NotImplementedError

    def _percentiles(self, scaled: np.ndarray, per_feature: np.ndarray) -> np.ndarray:
        """Per-column percentiles ``(n, n_features)``; NaN stays NaN."""
        raise NotImplementedError

    def _fit_columns(self, scaled: np.ndarray, columns: list[str]) -> None:
        """Learn each column's reference distribution from the scaled reference."""
        raise NotImplementedError

    def _save_columns(self, folder: pathlib.Path) -> dict[str, Any]:
        """Write the column distributions; return extra ``state.json`` entries."""
        raise NotImplementedError

    def _load_columns(self, folder: pathlib.Path, state: dict[str, Any]) -> None:
        """Read the column distributions written by :meth:`_save_columns`."""
        raise NotImplementedError

    def _has_columns(self) -> bool:
        """Whether the column distributions are present."""
        raise NotImplementedError

    # -- fit / run ----------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        *,
        eos_id: str | None = None,
        version: str | None = None,
        shared: SharedFitState | None = None,
    ):
        """Learn the per-column and whole-model reference distributions.

        Parameters
        ----------
        reference : pandas.DataFrame
            Predictions on the reference library.
        eos_id, version : str, optional
            Model id and version, needed only to fit ``shared`` here.
        shared : SharedFitState, optional
            Pre-fit shared state (as passed by :class:`ErsiliaQuality`).

        Returns
        -------
        PercentileScore
            ``self``, fitted.
        """
        t0 = time.perf_counter()
        name = type(self).__name__
        shared = _resolve_shared(
            reference, shared=shared, eos_id=eos_id, version=version, component=name
        )
        scaled = _reference_repr(shared, reference)
        self._fit_columns(scaled, list(shared.selected_columns))
        aggregate = _aggregate_percentiles(
            self._percentiles(scaled, self._per_feature(scaled))
        )
        self._sorted_self_aggregates = _sorted_finite(aggregate, name)
        self._anchor = float(
            np.nanmean(_cdf_score(aggregate, self._sorted_self_aggregates))
        )
        self._shared = shared
        self._finish_fit(t0)
        logger.debug(
            f"{name} fit | anchor={self._anchor:.4f}"
            f" | duration={self._fit_duration_seconds:.3f}s"
        )
        return self

    def run(
        self, query: pd.DataFrame, *, query_repr: np.ndarray | None = None
    ) -> PercentileRunResult:
        """Score query samples.

        Parameters
        ----------
        query : pandas.DataFrame
            The reference's numeric output columns.
        query_repr : numpy.ndarray, optional
            Pre-scaled, feature-selected query array; skips validation and scaling
            (the orchestrator shares this work across scores).

        Returns
        -------
        PercentileRunResult
            Percentile, Q66 aggregate, per-feature values and metadata.
        """
        self._check_fitted()
        assert self._shared is not None and self._sorted_self_aggregates is not None
        if query_repr is None:
            validate_against_schema(query, self._shared.schema)
            query_repr = _make_query_repr(self._shared, query)
        per_feature = self._per_feature(query_repr)
        per_feature_pct = self._percentiles(query_repr, per_feature)
        score = _cdf_score(
            _aggregate_percentiles(per_feature_pct), self._sorted_self_aggregates
        )
        idx = list(query.index)
        columns = list(self._shared.selected_columns)
        return PercentileRunResult(
            score=pd.Series(score, index=idx, name=f"{self.SCORE}_pct"),
            score_raw=pd.Series(
                _nan_aggregate(per_feature), index=idx, name=f"{self.SCORE}_raw"
            ),
            per_feature=pd.DataFrame(per_feature, index=idx, columns=columns),
            per_feature_pct=pd.DataFrame(per_feature_pct, index=idx, columns=columns),
            metadata={"anchor": self._anchor},
        )

    # -- persistence --------------------------------------------------------

    def _save_own(self, folder: pathlib.Path) -> None:
        assert self._shared is not None and self._sorted_self_aggregates is not None
        np.save(folder / SELF_AGGREGATES_FILE, self._sorted_self_aggregates)
        state = {
            "anchor": self._anchor,
            "columns": list(self._shared.selected_columns),
            **self._save_columns(folder),
        }
        with open(folder / STATE_FILE, "w") as f:
            json.dump(state, f)

    def _load_own(self, folder: pathlib.Path) -> None:
        assert self._shared is not None
        state = read_json(folder / STATE_FILE, self.NAME)
        if state["columns"] != list(self._shared.selected_columns):
            raise ValueError(
                f"{self.NAME}/{STATE_FILE} columns do not match the shared "
                "selected columns."
            )
        self._sorted_self_aggregates = np.load(
            require_file(folder / SELF_AGGREGATES_FILE, self.NAME)
        )
        self._anchor = float(state["anchor"])
        self._load_columns(folder, state)

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        return (
            self._shared is not None
            and self._sorted_self_aggregates is not None
            and self._anchor is not None
            and self._has_columns()
        )

    @property
    def anchor_(self) -> float:
        """Mean calibrated score of the reference molecules (about 0.5).

        Returns
        -------
        float
        """
        self._check_fitted()
        assert self._anchor is not None
        return self._anchor
