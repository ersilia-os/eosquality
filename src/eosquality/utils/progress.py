"""Shared ``rich`` progress bar for long per-molecule loops (stderr)."""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_console = Console(stderr=True, highlight=False)


def make_progress(label: str) -> Progress:
    """Return a not-yet-started progress bar titled ``label``.

    Parameters
    ----------
    label : str
        Title shown next to the bar.

    Returns
    -------
    rich.progress.Progress
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold cyan]{label}[/bold cyan]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    )
