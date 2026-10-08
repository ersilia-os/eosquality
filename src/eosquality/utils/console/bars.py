"""Progress bars."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from eosquality.utils.console._core import active_color, console, enabled


def track(items, label: str):
    """Iterate ``items``, with a progress bar when there are at least two.

    Silent (no bar) while the curated output is off or stderr is not a
    terminal, like :func:`progress`.

    Parameters
    ----------
    items : sequence
        What to iterate; a sequence, so its length is known.
    label : str
        What is being processed.

    Yields
    ------
    object
        Each item.
    """
    items = list(items)
    if len(items) < 2:
        yield from items
        return
    with progress(label) as bar:
        task = bar.add_task(label, total=len(items))
        for item in items:
            yield item
            bar.advance(task)


def progress(label: str) -> Progress:
    """A not-yet-started progress bar on the shared console.

    Disabled (draws nothing) while the curated output is off, so library
    calls stay silent, and when stderr is not a terminal (piped or CI
    output), where a transient bar cannot be erased and would leave blank
    lines; the step lines still record progress there.

    Parameters
    ----------
    label : str
        What is being processed.

    Returns
    -------
    rich.progress.Progress
    """
    return Progress(
        SpinnerColumn(style=active_color()),
        TextColumn(f"[{active_color()}]{label}[/]"),
        BarColumn(bar_width=None, complete_style=active_color()),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/]"),
        TimeElapsedColumn(),
        TextColumn("[dim]·[/]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
        disable=not (enabled() and console.is_terminal),
    )
