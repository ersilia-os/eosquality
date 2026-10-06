"""Diagnostic logging for eosquality: loguru on Rich, in the ZairaChem / Olinda style.

Two layers, as in ZairaChem and Olinda. User-facing status (steps, panels,
progress) belongs to :mod:`eosquality.utils.console`. This module is the
other half: DEBUG/INFO diagnostics for the record, and warnings and errors
that surface. The terminal sink is a ``RichHandler`` on the **same** shared
console, so a warning printed during a live progress bar is drawn cleanly
above it.

- **Terminal.** WARNING and above by default; DEBUG with
  :meth:`Logger.set_verbosity` (``-v`` in the CLI,
  ``ErsiliaQuality(verbose=True)``).
- **Log file.** :meth:`Logger.log_file` adds a DEBUG sink with
  ``module:function:line`` context. It belongs to one command, so it is
  neither rotated nor pruned (retention globbing could delete unrelated logs).
  An exception escaping the block is recorded in it, with its traceback.
  ``diagnose=False`` keeps variable values (e.g. SMILES) out of tracebacks.
  The CLI writes one per command (``<artifacts>/eosquality.log`` for
  ``fit``, ``<output>.log`` for ``run``).

eosquality logs through a loguru logger bound with ``extra["eosquality"]``,
and its sinks accept only those records. The standard-library loggers of
dependencies that print on their own (``eosframes``, which logs at INFO by
default) are routed into it, so their messages follow the same policy:
in the log file, and on screen only when they are warnings or ``-v`` is on. Handlers the host application added
are left untouched; only loguru's pristine default handler (id 0) is
removed, so package messages are not printed twice.
"""

from __future__ import annotations

import logging as _stdlib_logging
import pathlib
from contextlib import contextmanager

from loguru import logger as _root_logger
from rich.logging import RichHandler

from eosquality.utils.console import console as _console
from eosquality.utils.console import enable as _enable_console

try:
    _root_logger.remove(0)  # loguru's default stderr handler, if still present
except ValueError:
    pass

_loguru = _root_logger.bind(eosquality=True)

DEFAULT_LEVEL = "WARNING"
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)


# Dependencies whose standard-library loggers are routed into eosquality's.
ROUTED_LOGGERS = ("eosframes",)


def _only_eosquality(record) -> bool:
    return bool(record["extra"].get("eosquality"))


def _terminal(record) -> bool:
    """Terminal sink filter: eosquality records not marked ``file_only``."""
    return _only_eosquality(record) and not record["extra"].get("file_only")


class _ToLoguru(_stdlib_logging.Handler):
    """Forward standard-library log records to the eosquality loguru logger."""

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        try:
            try:
                level = _loguru.level(record.levelname).name
            except ValueError:
                level = record.levelno
            origin = {
                "name": record.name,
                "function": record.funcName,
                "line": record.lineno,
            }
            _loguru.patch(lambda r: r.update(origin)).opt(
                exception=record.exc_info
            ).log(level, record.getMessage())
        except Exception:  # logging must never break the caller
            self.handleError(record)


def _route_dependency_loggers() -> None:
    """Forward ``ROUTED_LOGGERS`` to loguru, unless the host already set them up.

    Done by logger name, without importing the dependency (eosframes pulls in
    pandas). eosframes only adds its own handler when its logger has none, so
    the forward installed here stays in place when it configures itself later.
    A logger that already has handlers was configured by the host application
    (or by the dependency itself, if imported first) and is left alone.
    """
    for name in ROUTED_LOGGERS:
        dependency = _stdlib_logging.getLogger(name)
        if dependency.handlers:
            continue
        dependency.handlers = [_ToLoguru()]
        dependency.setLevel(_stdlib_logging.DEBUG)
        dependency.propagate = False


class Logger:
    """The usual levels plus ``success()``, following the Ersilia convention."""

    def __init__(self) -> None:
        self.logger = _loguru
        self._sink_id: int | None = None
        self._verbose = False
        self.set_level(DEFAULT_LEVEL)
        _route_dependency_loggers()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_level(self, level: str) -> None:
        """Set the minimum level of the terminal sink.

        Parameters
        ----------
        level : str
            loguru level name.
        """
        if self._sink_id is not None:
            self.logger.remove(self._sink_id)
        self._level = level
        handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            markup=False,
            log_time_format="%H:%M:%S",
            show_path=False,
            show_level=True,
        )
        self._sink_id = self.logger.add(
            handler, format="{message}", level=level, filter=_terminal
        )

    @property
    def level(self) -> str:
        """Minimum level of the terminal sink.

        Returns
        -------
        str
        """
        return self._level

    @property
    def verbose(self) -> bool:
        """Whether DEBUG output is on.

        Returns
        -------
        bool
        """
        return self._verbose

    def set_verbosity(self, verbose: bool) -> None:
        """Toggle DEBUG terminal output (and the curated console output).

        ``verbose=True`` also turns the curated output of
        :mod:`eosquality.utils.console` on, so library users see the same
        steps as the CLI; ``verbose=False`` turns both off again (warnings
        only).

        Parameters
        ----------
        verbose : bool
            Turn DEBUG output on or off.
        """
        self._verbose = bool(verbose)
        self.set_level("DEBUG" if verbose else DEFAULT_LEVEL)
        _enable_console(self._verbose)

    def add_file(self, path: str | pathlib.Path) -> int:
        """Start writing every eosquality record (DEBUG+) to ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            Log file; appended to if it exists.

        Returns
        -------
        int
            Sink id, for :meth:`remove_file`.
        """
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        return self.logger.add(
            str(path),
            level="DEBUG",
            format=_FILE_FORMAT,
            backtrace=True,
            diagnose=False,
            filter=_only_eosquality,
            enqueue=False,
        )

    def remove_file(self, sink_id: int) -> None:
        """Stop and close a file sink added by :meth:`add_file`.

        Parameters
        ----------
        sink_id : int
            Id returned by :meth:`add_file`.
        """
        try:
            self.logger.remove(sink_id)
        except ValueError:
            pass

    @contextmanager
    def log_file(self, path: str | pathlib.Path):
        """Write a DEBUG log file for the duration of the block.

        Parameters
        ----------
        path : str or pathlib.Path
            Log file.

        Yields
        ------
        pathlib.Path
            The log file path.
        """
        sink = self.add_file(path)
        try:
            yield pathlib.Path(path)
        except BaseException as exc:
            # Record the failure while the sink is still open; the caller
            # (e.g. the CLI error handler) runs only after it is closed.
            self.logger.bind(file_only=True).opt(exception=True).error(
                f"failed: {type(exc).__name__}: {exc}"
            )
            raise
        finally:
            self.remove_file(sink)

    # ------------------------------------------------------------------
    # Levels
    # ------------------------------------------------------------------

    def debug(self, text: str) -> None:
        """Record ``text`` for diagnosis (terminal only with ``-v``).

        Parameters
        ----------
        text : str
            Message.
        """
        self.logger.opt(depth=1).debug(text)

    def info(self, text: str) -> None:
        """Record ``text`` (terminal only with ``-v``; user-facing status is the console's job).

        Parameters
        ----------
        text : str
            Message.
        """
        self.logger.opt(depth=1).info(text)

    def success(self, text: str) -> None:
        """Record the completion of a step (terminal only with ``-v``).

        Parameters
        ----------
        text : str
            Message.
        """
        self.logger.opt(depth=1).success(text)

    def warning(self, text: str) -> None:
        """Report something suspect that did not stop the run.

        Parameters
        ----------
        text : str
            Message.
        """
        self.logger.opt(depth=1).warning(text)

    def error(self, text: str) -> None:
        """Report a failure the caller is expected to handle or surface.

        Parameters
        ----------
        text : str
            Message.
        """
        self.logger.opt(depth=1).error(text)

    def exception(self, text: str) -> None:
        """Log an error with the active exception's traceback (in the log file).

        Parameters
        ----------
        text : str
            Message.
        """
        self.logger.opt(depth=1, exception=True).error(text)


logger = Logger()
