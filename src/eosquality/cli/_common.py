"""Helpers shared by the CLI commands: curated output, log files, failure handling."""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager

import click

from eosquality import set_verbosity
from eosquality.utils import console, parallel
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


def run_command(fn, *, verbose: bool, command: str, jobs: int | None = None) -> None:
    """Run a command body with curated output; failures become ``✖ error:`` lines.

    Turns the curated console on in the command's accent colour, and restores
    the previous console and verbosity state on exit. With ``verbose``, DEBUG
    logs and full tracebacks also go to the terminal; otherwise tracebacks
    only reach the command's log file (see ``Logger.log_file``).

    Parameters
    ----------
    fn : callable
        Zero-argument function holding the command's work.
    verbose : bool
        Enable DEBUG logging and on-screen tracebacks.
    command : str
        Command name, for its accent colour (``console.STEP_COLORS``).
    jobs : int, optional
        Worker processes for the per-molecule RDKit work (``-1``: every core).
        The CLI is a proper entry point, so it may start a pool (see
        :mod:`eosquality.utils.parallel`).
    """
    # Global output state is restored on exit, so calling the CLI in-process
    # (tests, notebooks) leaves library use silent again.
    previous = (console.enabled(), console.active_color(), logger.verbose, logger.level)
    console.enable(True)
    console.set_active_color(console.STEP_COLORS.get(command, "cyan"))
    if verbose:
        set_verbosity(True)
    ctx = click.get_current_context()
    try:
        with parallel.workers(jobs):
            fn()
    except CliError as exc:
        console.echo(f"[bold red]error:[/] {console.plain(exc)}", "error")
        ctx.exit(1)
    except Exception as exc:  # anything unexpected still exits cleanly with status 1
        console.echo(f"[bold red]error:[/] {console.plain(exc)}", "error")
        if verbose:
            console.console.print_exception()
        ctx.exit(1)
    finally:
        set_verbosity(previous[2])
        logger.set_level(previous[3])
        console.enable(previous[0])
        console.set_active_color(previous[1])


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


jobs_option = click.option(
    "--jobs",
    "-j",
    default=-1,
    show_default=True,
    metavar="N",
    help="Worker processes for the RDKit descriptors (-1: every core, up to 16; 1: none).",
)

verbose_option = click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show debug messages and tracebacks.",
)
