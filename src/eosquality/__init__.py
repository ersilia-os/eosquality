"""eosquality: assess the quality of query data against a fitted reference population."""

import importlib.metadata as _importlib_metadata

from packaging.version import Version as _Version

from eosquality._registry import ALL_SCORES
from eosquality.library.identity import LIBRARY_ID, library_major
from eosquality.utils.logging import logger as _logger

# The public classes are resolved on first access (PEP 562): importing them
# eagerly pulls in pandas, scikit-learn, SciPy and RDKit (about 2 s), which
# the CLI must not pay just to parse its arguments.
_LAZY = {
    "ErsiliaQuality": "eosquality.quality",
    "RunResult": "eosquality.results",
    **{
        name: "eosquality.scores"
        for name in (
            "Consistency",
            "ConsistencyRunResult",
            "Extremity",
            "ExtremityRunResult",
            "Signal",
            "SignalRunResult",
            "Support",
            "SupportRunResult",
            "Typicality",
            "TypicalityRunResult",
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


def _check_library_matches_package_major() -> None:
    """Fail loudly at import if library_vN and package major X have drifted.

    Policy: ``eosquality X.y.z`` ships exactly one library,
    ``ersilia_reference_library_vX``. Any release where these disagree is a
    packaging bug that must not escape CI.
    """
    try:
        pkg_version = _importlib_metadata.version("eosquality")
    except _importlib_metadata.PackageNotFoundError:
        return
    pkg_major = _Version(pkg_version).major
    lib_major = library_major()
    if pkg_major != lib_major:
        raise RuntimeError(
            f"Reference library / package version mismatch: "
            f"eosquality is {pkg_version} (major={pkg_major}) but "
            f"LIBRARY_ID={LIBRARY_ID!r} (major={lib_major}). "
            "These must move together. This is a release-engineering bug."
        )


_check_library_matches_package_major()


def set_verbosity(verbose: bool) -> None:
    """Turn the step-by-step output and DEBUG logs on, or back to warnings only.

    Parameters
    ----------
    verbose : bool
        ``True`` for the curated step output (as in the CLI) and DEBUG
        messages; ``False`` for warnings only.
    """
    _logger.set_verbosity(verbose)


def set_log_level(level: str) -> None:
    """Set the package log level (e.g. ``"INFO"`` to see progress messages).

    Parameters
    ----------
    level : str
        A loguru level name such as ``"DEBUG"``, ``"INFO"`` or ``"WARNING"``.
    """
    _logger.set_level(level)


__all__ = [
    "ALL_SCORES",
    "ErsiliaQuality",
    "RunResult",
    "Typicality",
    "TypicalityRunResult",
    "Support",
    "SupportRunResult",
    "Consistency",
    "ConsistencyRunResult",
    "Extremity",
    "ExtremityRunResult",
    "Signal",
    "SignalRunResult",
    "set_log_level",
    "set_verbosity",
]
