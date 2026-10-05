"""Score names, canonical order and component classes shared by the orchestrator."""

from eosquality.scores.consistency import Consistency
from eosquality.scores.extremity import Extremity
from eosquality.scores.signal import Signal
from eosquality.scores.support import Support
from eosquality.scores.typicality import Typicality

MIN_REFERENCE_SAMPLES = 10_000

DEFAULT_SCORES: tuple[str, ...] = (
    "typicality",
    "support",
    "consistency",
    "extremity",
)
# All valid score names, including the opt-in ones that are not in DEFAULT_SCORES.
# Signal is opt-in (provisional): users must pass ``scores=DEFAULT_SCORES + ("signal",)``
# or similar to enable it. Validation uses this set, not DEFAULT_SCORES.
ALL_SCORES: tuple[str, ...] = DEFAULT_SCORES + ("signal",)
# Canonical component order: fit, run, save, output columns and metadata.
SCORE_ORDER: tuple[str, ...] = (
    "typicality",
    "extremity",
    "support",
    "consistency",
    "signal",
)
SCORE_CLASSES = {
    "typicality": Typicality,
    "extremity": Extremity,
    "support": Support,
    "consistency": Consistency,
    "signal": Signal,
}
# Scores that need the vector index at fit time (Signal reads the library's
# descriptor matrices from the index folder; it never queries the index).
INDEX_AWARE = frozenset({"support", "consistency", "signal"})
KNN_USERS = frozenset({"support", "consistency"})
# Training-modality components (present iff training sets were fit).
TRAINING_ORDER: tuple[str, ...] = ("training_distance", "training_difficulty")
