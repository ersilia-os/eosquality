"""Helpers shared by the CLI commands: user-facing messages and failure handling."""

from __future__ import annotations

import os
import traceback

import click

from eosquality import set_verbosity


class CliError(Exception):
    """A user-facing error: printed as ``error: <message>`` and exit status 1."""


def say(message: str, *, err: bool = True) -> None:
    """Print a progress or result line (stderr by default).

    Parameters
    ----------
    message : str
        Text to print.
    err : bool, optional
        Print to stderr (default) instead of stdout.
    """
    click.echo(message, err=err)


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


def run_command(fn, *, verbose: bool) -> None:
    """Run a command body, turning failures into ``error:`` lines and exit codes.

    Parameters
    ----------
    fn : callable
        Zero-argument function holding the command's work.
    verbose : bool
        Enable DEBUG logging and print full tracebacks on failure.
    """
    if verbose:
        set_verbosity(True)
    ctx = click.get_current_context()
    try:
        fn()
    except CliError as exc:
        say(f"error: {exc}")
        ctx.exit(1)
    except Exception as exc:  # anything unexpected still exits cleanly with status 1
        say(f"error: {exc}")
        if verbose:
            traceback.print_exc()
        ctx.exit(1)


verbose_option = click.option(
    "--verbose", "-v", is_flag=True, help="Print debug messages and diagnostic tables."
)
