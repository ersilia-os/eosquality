"""Score names, canonical order and constants shared by the orchestrator and the CLI.

Constants only, with no heavy imports: the CLI reads them to build its help
text, so importing this module must stay cheap (see ``tests/test_startup.py``).
The component classes are mapped in :mod:`eosquality._artifacts`.

Every score has an internal **component** name (the attribute on
:class:`~eosquality.quality.ErsiliaQuality` and the artifacts subfolder) and a
public **score** name, prefixed by modality: ``ref_`` for scores fitted on
the reference library predictions, ``trn_`` for scores fitted on the
training sets. Public names are the output columns and the values accepted
by ``exclude`` (``--exclude`` in the CLI).
"""

# Nearest neighbours per molecule (FP kNN of support and consistency).
N_NEIGHBORS = 5
# Feature-selection cap of the shared state (``fit_shared``).
DEFAULT_MAX_FEATURES = 10

# Canonical component order: fit, run, save, output columns and metadata.
SCORE_ORDER: tuple[str, ...] = (
    "typicality",
    "extremity",
    "support",
    "consistency",
    "signal",
)
# Training-modality components (present iff training sets were fit).
TRAINING_ORDER: tuple[str, ...] = (
    "training_distance",
    "training_physchem",
    "training_match",
)

# Component → public score name (and output column prefix).
SCORE_NAMES: dict[str, str] = {
    "typicality": "ref_typicality",
    "extremity": "ref_extremity",
    "support": "ref_support",
    "consistency": "ref_consistency",
    "signal": "ref_signal",
    "training_distance": "trn_tanimoto",
    "training_physchem": "trn_physchem",
    "training_match": "trn_match",
}
# Scores that emit a second output column of their own, for display: trn_match
# also writes trn_scaffold.
ALSO_EMITS: dict[str, tuple[str, ...]] = {"trn_match": ("trn_scaffold",)}
COMPONENTS: dict[str, str] = {score: comp for comp, score in SCORE_NAMES.items()}
REFERENCE_SCORES: tuple[str, ...] = tuple(SCORE_NAMES[c] for c in SCORE_ORDER)
TRAINING_SCORES: tuple[str, ...] = tuple(SCORE_NAMES[c] for c in TRAINING_ORDER)
ALL_SCORES: tuple[str, ...] = REFERENCE_SCORES + TRAINING_SCORES

# Scores that need the vector index at fit time (Signal reads the library's
# descriptor matrices from the index folder; it never queries the index).
INDEX_AWARE = frozenset({"support", "consistency", "signal"})
KNN_USERS = frozenset({"support", "consistency"})


def score_name(component: str) -> str:
    """Public, modality-prefixed score name of a component.

    Parameters
    ----------
    component : str
        Internal component name, e.g. ``"support"``.

    Returns
    -------
    str
        e.g. ``"ref_support"``.
    """
    return SCORE_NAMES[component]


def split_exclude(exclude) -> tuple[set[str], set[str]]:
    """Validate excluded score names; return the excluded components per modality.

    Parameters
    ----------
    exclude : iterable of str
        Public score names (``ALL_SCORES``).

    Returns
    -------
    tuple of (set of str, set of str)
        Excluded reference components and excluded training components.

    Raises
    ------
    ValueError
        If a name is not a score.
    """
    names = set(exclude or ())
    unknown = sorted(names - set(ALL_SCORES))
    if unknown:
        raise ValueError(
            f"Unknown score(s) {unknown}; choose from: {', '.join(ALL_SCORES)}."
        )
    components = {COMPONENTS[n] for n in names}
    return components & set(SCORE_ORDER), components & set(TRAINING_ORDER)
