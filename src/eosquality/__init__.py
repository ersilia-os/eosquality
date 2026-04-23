"""eosquality: assess the quality of query data against a fitted reference population."""

import importlib.metadata as _importlib_metadata

from packaging.version import Version as _Version

from eosquality._library import LIBRARY_ID, library_major
from eosquality.quality.api import ErsiliaQuality
from eosquality.scoring.run import RunResult
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
        # Running from an un-installed checkout (e.g. `python -c 'import eosquality'`
        # with PYTHONPATH pointing at src/). Skip — the pytest guardrail covers this.
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
    """Enable or disable informative log output globally.

    Parameters
    ----------
    verbose:
        ``True`` to enable progress logs and diagnostic tables.
        ``False`` (default) to suppress all output.
    """
    _logger.set_verbosity(verbose)


__all__ = ["ErsiliaQuality", "RunResult", "set_verbosity"]
