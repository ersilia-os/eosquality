"""Sections and numbered steps."""

from __future__ import annotations

import time
from contextlib import contextmanager

from eosquality.utils.console._core import (
    active_color,
    console,
    echo,
    enabled,
    rule,
    set_active_color,
)
from eosquality.utils.console.fmt import elapsed, resources


class _Handle:
    """Mutable handle yielded by :func:`section` and :class:`Steps`."""

    summary: str | None = None


@contextmanager
def section(title: str, *, color: str | None = None):
    """A themed section: a rule, then a timed closing line in the accent colour.

    Sets the accent colour for everything printed inside. Set
    ``handle.summary`` to put a short result on the closing line.

    Parameters
    ----------
    title : str
        Section title.
    color : str, optional
        Accent colour for the section (default: the active colour).

    Yields
    ------
    _Handle
    """
    previous = active_color()
    color = color or previous
    set_active_color(color)
    if enabled():
        console.print()  # a blank line between sections
    rule(title, style=color, right=resources())
    handle = _Handle()
    started = time.perf_counter()
    ok = True
    try:
        yield handle
    except BaseException:
        ok = False
        raise
    finally:
        took = elapsed(time.perf_counter() - started)
        tail = f"{handle.summary} · {took}" if handle.summary else took
        if enabled():
            glyph, gcolor = ("✓", color) if ok else ("✕", "red")
            console.print(f"  [{gcolor}]{glyph}[/] [dim]{tail}[/]")
        set_active_color(previous)


class Steps:
    """Numbered steps of a section: ``▪ Step i/N · title`` … ``✓ summary · 1.2s``.

    Parameters
    ----------
    total : int
        Number of steps announced in the headers.

    Examples
    --------
    >>> steps = Steps(2)
    >>> with steps("Load") as st:
    ...     st.summary = "3 files"
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0

    @contextmanager
    def __call__(self, title: str):
        """Run one step.

        Parameters
        ----------
        title : str
            Step title.

        Yields
        ------
        _Handle
            Set ``summary`` for the closing line.
        """
        self.index += 1
        echo(f"[bold]Step {self.index}/{self.total}[/] · {title}", "run")
        handle = _Handle()
        started = time.perf_counter()
        ok = True
        try:
            yield handle
        except BaseException:
            ok = False
            raise
        finally:
            took = elapsed(time.perf_counter() - started, precise=True)
            if not ok:
                echo(f"[red]failed[/] [dim]· {took}[/]", "error", sub=True)
            elif handle.summary is not None:
                echo(f"{handle.summary} [dim]· {took}[/]", "success", sub=True)
            else:
                echo(f"[dim]{took}[/]", "success", sub=True)
