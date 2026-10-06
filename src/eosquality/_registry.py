"""Score names, canonical order and defaults shared by the orchestrator and the CLI.

Constants only, with no heavy imports: the CLI reads them to build its help
text, so importing this module must stay cheap (see ``tests/test_startup.py``).
The component classes are mapped in :mod:`eosquality._artifacts`.
"""

MIN_REFERENCE_SAMPLES = 10_000
# Feature-selection cap of the shared state (``fit_shared``).
DEFAULT_MAX_FEATURES = 10

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
# Scores that need the vector index at fit time (Signal reads the library's
# descriptor matrices from the index folder; it never queries the index).
INDEX_AWARE = frozenset({"support", "consistency", "signal"})
KNN_USERS = frozenset({"support", "consistency"})
# Training-modality components (present iff training sets were fit).
TRAINING_ORDER: tuple[str, ...] = ("training_distance", "training_difficulty")
