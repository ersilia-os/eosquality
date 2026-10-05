"""Helpers shared by the CLI commands: curated output, log files, failure handling."""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager

import click

from eosquality import set_verbosity
from eosquality.utils import console
from eosquality.utils.logging import logger


class CliError(Exception):
    """A user-facing error: printed as ``✖ error: <message>`` and exit status 1."""


def require_new_path(path: str, what: str = "output path") -> None:
    """Raise :class:`CliError` if ``path`` already exists.

    Parameters
    ----------
    path : str
        Path that must not exist yet.
    what : str, optional
        How to name the path in the error message.
    """
    if os.path.exists(path):
        raise CliError(
            f"{what} '{path}' already exists; delete or move it before re-running."
        )


def run_command(fn, *, verbose: bool, command: str) -> None:
    """Run a command body with curated output; failures become ``✖ error:`` lines.

    Turns the curated console on in the command's accent colour. With
    ``verbose``, DEBUG logs and full tracebacks also go to the terminal;
    otherwise tracebacks only reach the log file, if the command opened one.

    Parameters
    ----------
    fn : callable
        Zero-argument function holding the command's work.
    verbose : bool
        Enable DEBUG logging and on-screen tracebacks.
    command : str
        Command name, for its accent colour (``console.STEP_COLORS``).
    """
    console.enable(True)
    console.set_active_color(console.STEP_COLORS.get(command, "cyan"))
    if verbose:
        set_verbosity(True)
    ctx = click.get_current_context()
    try:
        fn()
    except CliError as exc:
        logger.debug(f"{command} | error: {exc}")
        console.echo(f"[bold red]error:[/] {exc}", "error")
        ctx.exit(1)
    except Exception as exc:  # anything unexpected still exits cleanly with status 1
        logger.logger.opt(exception=True).debug(f"{command} | unexpected error")
        console.echo(f"[bold red]error:[/] {exc}", "error")
        if verbose:
            console.console.print_exception()
        ctx.exit(1)


@contextmanager
def staged_log(final_path: pathlib.Path):
    """Log to a temporary file, then append it to ``final_path`` on success.

    For ``fit``, whose artifacts folder only exists once saved. On failure
    the temporary log is kept and its path printed.

    Parameters
    ----------
    final_path : pathlib.Path
        Where the log belongs once the command succeeds.

    Yields
    ------
    pathlib.Path
        ``final_path``.
    """
    handle, tmp = tempfile.mkstemp(prefix="eosquality_", suffix=".log")
    os.close(handle)
    ok = False
    try:
        with logger.log_file(tmp):
            yield final_path
        ok = True
    finally:
        if ok and final_path.parent.is_dir():
            with open(final_path, "a") as out, open(tmp) as src:
                shutil.copyfileobj(src, out)
            os.remove(tmp)
        else:
            console.echo(f"[dim]log →[/] {console.path(tmp)}")


verbose_option = click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Also print debug messages and full tracebacks on screen.",
)
