"""eosquality: assess the quality of query data against a fitted reference population."""

import importlib.metadata as _importlib_metadata

from packaging.version import Version as _Version

from eosquality.library.identity import LIBRARY_ID, library_major
from eosquality.quality import ALL_SCORES, DEFAULT_SCORES, ErsiliaQuality, RunResult
from eosquality.scores import (
    Consistency,
    ConsistencyRunResult,
    Extremity,
    ExtremityRunResult,
    Signal,
    SignalRunResult,
    Support,
    SupportRunResult,
    Typicality,
    TypicalityRunResult,
)
from eosquality.utils.logging import logger as _logger


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
    """Enable (DEBUG + diagnostic tables) or disable (WARNING only) log output."""
    _logger.set_verbosity(verbose)


def set_log_level(level: str) -> None:
    """Set the package log level (e.g. ``"INFO"`` to see progress messages)."""
    _logger.set_level(level)


__all__ = [
    "ALL_SCORES",
    "DEFAULT_SCORES",
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
