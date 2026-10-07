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
    ``trn_tanimoto_pct``, ``trn_tanimoto_raw`` (one whole-model value: Q66
    across output columns of the mean distance to the 5 nearest training
    molecules, and its percentile among the training set's own
    leave-one-out values), ``trn_physchem_pct`` (the same
    percentile in physchem space), ``trn_physchem_raw`` (a similarity,
    ``1 - d / 18.70``: 1 is identical, 0 no closer than a random library
    pair, unclipped; the distance is in ``training_details``), ``trn_match`` and ``trn_scaffold``
    (1 / 0: the InChIKey connectivity layer of the molecule, or of its Murcko
    scaffold, is in a training set; the scaffold flag is NA without one), and
    ``trn_difficulty`` when the error model is included.

    ``metadata`` has ``n_reference`` (reference modality) plus each score's
    run metadata with keys prefixed by the score name (e.g.
    ``ref_support_k``, ``ref_consistency_n_fp_bins``,
    ``trn_tanimoto_n_columns``).

    ``training_details`` (training modality only) has one row per query:
    the whole-model distances and the 5 nearest training molecules over all
    output columns, with their similarities and columns.
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]
    training_details: pd.DataFrame | None = None
