"""Result container returned by :meth:`ErsiliaQuality.run <eosquality.quality.ErsiliaQuality.run>`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    ``scores`` is a per-query DataFrame with the columns of each fitted
    score, in canonical order: ``ref_typicality_pct``, ``ref_typicality_raw``,
    ``ref_extremity_pct``, ``ref_extremity_raw``, ``ref_support``,
    ``ref_support_raw``, ``ref_support_log``, ``ref_consistency``,
    ``ref_consistency_raw``, ``ref_signal``, ``ref_signal_raw``. The
    calibrated column is in ``(0, 1]``; ``*_raw`` is the pre-calibration
    value; ``ref_support_log = −log10(ref_support)``. Scores that were not
    fit are absent.

    Training-modality columns (when training sets were fit) follow:
    ``trn_tanimoto_pct`` (one whole-model value: the similarity percentile of
    the mean Tanimoto distance to the 5 nearest training molecules among the
    training set's own leave-one-out values, taken at the 66th percentile
    across output columns; higher is closer), ``trn_tanimoto_raw`` (the mean
    Tanimoto similarity at the same point, higher is closer),
    ``trn_physchem_pct`` (the same percentile in physchem space, higher is
    closer), ``trn_physchem_raw`` (a similarity,
    ``1 - d / 18.70``: 1 is identical, 0 no closer than a random library
    pair, unclipped; the distance is in ``training_details``), ``trn_match`` and ``trn_scaffold``
    (1 / 0: the InChIKey connectivity layer of the molecule, or of its Murcko
    scaffold, is in a training set; the scaffold flag is NA without one).

    ``metadata`` has ``n_reference`` (reference modality) plus each score's
    run metadata with keys prefixed by the score name (e.g.
    ``ref_support_k``, ``ref_consistency_n_fp_bins``,
    ``trn_tanimoto_n_columns``).

    ``training_details`` (training modality only) has one row per query:
    the whole-model distances and the 5 nearest training molecules over all
    output columns, with their similarities and columns.

    ``reference_details`` (reference modality with typicality or extremity
    fitted) has one row per query: ``key``, ``input`` and, for every selected
    output column and each of those scores, ``<column>_typicality_raw``
    (``count / max count``) or ``<column>_extremity_raw``
    (``min(|scaled|, 1)``) and the matching ``_pct`` (its percentile among
    the reference library's values of that column).
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]
    training_details: pd.DataFrame | None = None
    reference_details: pd.DataFrame | None = None
