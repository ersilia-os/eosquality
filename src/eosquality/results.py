"""Result container returned by :meth:`ErsiliaQuality.run <eosquality.quality.ErsiliaQuality.run>`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    ``scores`` is a per-query DataFrame with the columns of each fitted
    component, in canonical order: ``typicality``, ``typicality_raw``,
    ``extremity``, ``extremity_raw``, ``support``, ``support_raw``,
    ``support_log``, ``consistency``, ``consistency_raw``, ``signal``,
    ``signal_raw``. The calibrated column is in ``(0, 1]``; ``*_raw`` is the
    pre-calibration value; ``support_log = −log10(support)``. Components
    that were not fit are absent.

    Training-modality columns (when training sets were fit) follow:
    ``training_distance``, ``training_distance_raw`` (one whole-model value:
    Q66 across output columns of the mean distance to the 5 nearest training
    molecules), ``in_training``.

    ``metadata`` has ``n_reference`` (reference modality) plus each
    component's run metadata with keys prefixed by the component name (e.g.
    ``support_k``, ``consistency_n_fp_bins``, ``training_distance_n_columns``).

    ``training_details`` (training modality only) has one row per query:
    the whole-model distances and the 5 nearest training molecules over all
    output columns, with their similarities and columns.
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]
    training_details: pd.DataFrame | None = None
