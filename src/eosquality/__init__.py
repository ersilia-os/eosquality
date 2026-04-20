"""eosquality: assess the quality of query data against a fitted reference population."""

from eosquality.quality.api import ErsiliaQuality
from eosquality.scoring.run import RunResult
from eosquality.utils.logging import logger as _logger


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
