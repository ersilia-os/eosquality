"""eosquality: assess the quality of query data against a fitted reference population."""

from eosquality._registry import ALL_SCORES

# The public classes are resolved on first access (PEP 562): importing them
# eagerly pulls in pandas, scikit-learn, SciPy and RDKit (about 2 s), which
# the CLI must not pay just to parse its arguments.
_LAZY = {
    "ErsiliaQuality": "eosquality.quality",
    "RunResult": "eosquality.results",
    **{
        name: "eosquality.scores"
        for name in (
            "Extremity",
            "PercentileRunResult",
            "ReferenceMatch",
            "ReferenceMatchRunResult",
            "Typicality",
        )
    },
}


def __getattr__(name: str):
    """Import a public class on first access.

    Parameters
    ----------
    name : str
        Attribute name.

    Returns
    -------
    object
        The class.

    Raises
    ------
    AttributeError
        If ``name`` is not a public attribute.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'eosquality' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Module attributes, including the lazily imported classes.

    Returns
    -------
    list of str
    """
    return sorted(set(globals()) | set(_LAZY))


def set_verbosity(verbose: bool) -> None:
    """Turn the step-by-step output and DEBUG logs on, or back to warnings only.

    Parameters
    ----------
    verbose : bool
        ``True`` for the curated step output (as in the CLI) and DEBUG
        messages; ``False`` for warnings only.
    """
    from eosquality.utils.logging import logger

    logger.set_verbosity(verbose)


def set_log_level(level: str) -> None:
    """Set the package log level (e.g. ``"INFO"`` to see progress messages).

    Parameters
    ----------
    level : str
        A loguru level name such as ``"DEBUG"``, ``"INFO"`` or ``"WARNING"``.
    """
    from eosquality.utils.logging import logger

    logger.set_level(level)


__all__ = [
    "ALL_SCORES",
    "ErsiliaQuality",
    "RunResult",
    "Typicality",
    "Extremity",
    "ReferenceMatch",
    "PercentileRunResult",
    "ReferenceMatchRunResult",
    "set_log_level",
    "set_verbosity",
]
