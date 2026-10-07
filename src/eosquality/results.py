"""Result container returned by :meth:`ErsiliaQuality.run <eosquality.quality.ErsiliaQuality.run>`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    Column and key definitions are in ``docs/api.md``.

    Attributes
    ----------
    scores : pandas.DataFrame
        One row per query, indexed like the query: the columns of each fitted
        score, reference scores first (``ref_typicality_pct`` / ``_raw``,
        ``ref_extremity_pct`` / ``_raw``, ``ref_match``, ``ref_scaffold``),
        then the training scores (``trn_tanimoto_pct`` / ``_raw``,
        ``trn_physchem_pct`` / ``_raw``, ``trn_match``, ``trn_scaffold``).
        Scores that were not fit are absent.
    metadata : dict
        ``n_reference`` plus each score's run metadata, keys prefixed by the
        score name (e.g. ``ref_typicality_anchor``).
    training_details : pandas.DataFrame or None
        Per query: the whole-model distances and the 5 nearest training
        molecules. ``None`` without the training modality.
    reference_details : pandas.DataFrame or None
        Per query: the per-column typicality and extremity, raw and percentile.
        ``None`` when neither score is fitted.
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]
    training_details: pd.DataFrame | None = None
    reference_details: pd.DataFrame | None = None
