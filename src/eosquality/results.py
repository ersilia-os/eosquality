"""Result container returned by :meth:`ErsiliaQuality.run <eosquality.quality.ErsiliaQuality.run>`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    ``scores`` is a per-query DataFrame with the columns of each fitted
    score, in canonical order: ``ref_typicality``, ``ref_typicality_raw``,
    ``ref_extremity``, ``ref_extremity_raw``, ``ref_support``,
    ``ref_support_raw``, ``ref_support_log``, ``ref_consistency``,
    ``ref_consistency_raw``, ``ref_signal``, ``ref_signal_raw``. The
    calibrated column is in ``(0, 1]``; ``*_raw`` is the pre-calibration
    value; ``ref_support_log = −log10(ref_support)``. Scores that were not
    fit are absent.

    Training-modality columns (when training sets were fit) follow:
    ``trn_distance``, ``trn_distance_raw`` (one whole-model value: Q66 across
    output columns of the mean distance to the 5 nearest training
    molecules), ``trn_difficulty``, ``trn_in_training``.

    ``metadata`` has ``n_reference`` (reference modality) plus each score's
    run metadata with keys prefixed by the score name (e.g.
    ``ref_support_k``, ``ref_consistency_n_fp_bins``,
    ``trn_distance_n_columns``).

    ``training_details`` (training modality only) has one row per query:
    the whole-model distances and the 5 nearest training molecules over all
    output columns, with their similarities and columns.
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]
    training_details: pd.DataFrame | None = None
