"""Training modality: per-output-column training sets and the state built from them.

Each output column of a model may come with its own training set (SMILES,
optionally labels ``y``). :func:`load_training` reads and standardises them;
:func:`fit_training` builds one small Morgan :class:`VectorIndex` per column;
training-aware scores (``scores/training_domain.py``) sit on top. Persisted
under ``training_mode/training_sets/`` and versioned separately from the reference
artifacts (``TRAINING_FORMAT_VERSION``).
"""

from eosquality.training.data import TrainingColumn, load_training
from eosquality.training.state import (
    TRAINING_FORMAT_VERSION,
    TrainingFitState,
    fit_training,
    load_training_state,
    save_training_state,
)

__all__ = [
    "TRAINING_FORMAT_VERSION",
    "TrainingColumn",
    "TrainingFitState",
    "fit_training",
    "load_training",
    "load_training_state",
    "save_training_state",
]
