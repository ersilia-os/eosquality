"""Curated terminal output for eosquality, in the Ersilia (ZairaChem / Olinda) style.

Two layers, as in ZairaChem and Olinda: this module prints the **curated,
user-facing** stream (themed section rules, numbered steps that close with a
timed ``✓`` line, borderless detail blocks, rounded summary panels, progress
bars); :mod:`eosquality.utils.logging` keeps the **diagnostic** stream
(loguru: warnings on screen, everything in the log file). Both write through
the one shared :data:`console`, so log lines and live progress interleave
cleanly.

As a library, eosquality is silent: nothing here prints until
:func:`enable` is called, which the CLI does for every command and
``ErsiliaQuality(verbose=True)`` / ``eosquality.set_verbosity(True)`` do too.

    from eosquality.utils import console

    with console.section("Reference modality"):
        steps = console.Steps(3)
        with steps("Shared state") as st:
            ...
            st.summary = "4 → 3 features"
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

#: The one console for the whole package (stderr keeps stdout free for data).
#: ``highlight=False``: all colour comes from explicit markup, not Rich's
#: number highlighter, which would speckle digits against the step palette.
console = Console(stderr=True, highlight=False)

#: One accent colour per command, so the terminal shifts hue as work moves on.
STEP_COLORS = {
    "fit": "bright_green",
    "run": "blue",
    "build": "cyan",
    "setup": "bright_cyan",
}

_ICONS = {
    "success": ("✓", "green"),
    "warning": ("⚠", "yellow"),
    "error": ("✖", "red"),
    "run": ("▪", "cyan"),
    "info": ("·", "dim"),
}

_enabled = False
_active_color = "cyan"


def enable(flag: bool = True) -> None:
    """Turn the curated output on (CLI, verbose library use) or off.

    Parameters
    ----------
    flag : bool, optional
        ``True`` to print, ``False`` to stay silent.
    """
    global _enabled
    _enabled = bool(flag)


def enabled() -> bool:
    """Whether the curated output is on.

    Returns
    -------
    bool
    """
    return _enabled


def set_active_color(color: str) -> None:
    """Set the accent colour used by rules, steps and panels.

    Parameters
    ----------
    color : str
        A Rich colour name.
    """
    global _active_color
    _active_color = color or "cyan"


def active_color() -> str:
    """The current accent colour.

    Returns
    -------
    str
    """
    return _active_color


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


def echo(text: str, kind: str = "info", *, sub: bool = False) -> None:
    """One status line with an Ersilia-style glyph: ``✓ ⚠ ✖ ▪ ·``.

    Parameters
    ----------
    text : str
        Message (Rich markup allowed).
    kind : {"success", "warning", "error", "run", "info"}, optional
        Glyph and colour.
    sub : bool, optional
        Indent one more level, for a line that elaborates on the one above.
    """
    if not _enabled:
        return
    icon, style = _ICONS.get(kind, _ICONS["info"])
    if kind == "run":
        style = _active_color
    console.print(f"{'    ' if sub else '  '}[{style}]{icon}[/] {text}")


def success(text: str) -> None:
    """A green ``✓`` line (use ``→`` in ``text`` for produced paths).

    Parameters
    ----------
    text : str
        Message.
    """
    echo(text, "success")


def rule(title: str, *, style: str | None = None, right: str | None = None) -> None:
    """Left-aligned themed section divider with an optional dim caption.

    Parameters
    ----------
    title : str
        Section title.
    style : str, optional
        Colour (default: the active colour).
    right : str, optional
        Dim caption after the title.
    """
    if not _enabled:
        return
    style = style or _active_color
    label = f"[bold {style}]{title}[/]"
    if right:
        label += f"   [dim]{right}[/]"
    console.rule(label, align="left", style=style)


# ---------------------------------------------------------------------------
# Sections and steps
# ---------------------------------------------------------------------------


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
    previous = _active_color
    color = color or previous
    set_active_color(color)
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
        if _enabled:
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

    def skip(self, title: str, reason: str) -> None:
        """Announce a step that does not run, and why.

        Parameters
        ----------
        title : str
            Step title.
        reason : str
            Why it is skipped.
        """
        self.index += 1
        echo(f"[bold]Step {self.index}/{self.total}[/] · {title}", "run")
        echo(f"[dim]skipped: {reason}[/]", "info", sub=True)


# ---------------------------------------------------------------------------
# Blocks and panels
# ---------------------------------------------------------------------------


def detail(rows, *, indent: int = 6) -> None:
    """Borderless block of dim right-aligned labels and values.

    Parameters
    ----------
    rows : iterable of (str, object)
        Label/value pairs.
    indent : int, optional
        Left padding.
    """
    if not _enabled:
        return
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(justify="left", overflow="fold")
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(Padding(table, (0, 0, 0, indent)))


def table(columns, rows, *, title: str | None = None, indent: int = 6) -> None:
    """Borderless table with a themed header row.

    Parameters
    ----------
    columns : sequence of str
        Header labels; the first column is left-aligned, the others right.
    rows : iterable of sequence
        Cell values.
    title : str, optional
        Dim title above the table.
    indent : int, optional
        Left padding.
    """
    if not _enabled:
        return
    out = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style=f"bold {_active_color}",
        title=f"[dim]{title}[/]" if title else None,
        title_justify="left",
    )
    for i, name in enumerate(columns):
        out.add_column(name, justify="left" if i == 0 else "right", no_wrap=i == 0)
    for row in rows:
        out.add_row(*(str(v) for v in row))
    console.print(Padding(out, (0, 0, 0, indent)))


def summary_panel(
    title: str, rows, *, color: str | None = None, icon: str = ""
) -> None:
    """Rounded, left-titled panel of key/value rows (run header, final summary).

    Parameters
    ----------
    title : str
        Panel title.
    rows : iterable of (str, object)
        Label/value pairs.
    color : str, optional
        Border and label colour (default: the active colour).
    icon : str, optional
        Glyph before the title.
    """
    if not _enabled:
        return
    color = color or _active_color
    grid = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    grid.add_column(justify="right", style=f"bold {color}", no_wrap=True)
    grid.add_column(justify="left", overflow="fold")
    for key, value in rows:
        grid.add_row(str(key), str(value))
    heading = f"{icon}  {title}" if icon else title
    console.print(
        Panel(
            grid,
            title=f"[bold {color}]{heading}[/]",
            title_align="left",
            border_style=color,
            box=box.ROUNDED,
            padding=(1, 2),
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


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
        SpinnerColumn(style=_active_color),
        TextColumn(f"[{_active_color}]{label}[/]"),
        BarColumn(bar_width=None, complete_style=_active_color),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/]"),
        TimeElapsedColumn(),
        TextColumn("[dim]·[/]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
        disable=not (_enabled and console.is_terminal),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def path(value, keep: int = 3) -> str:
    """Render a path compactly: ``~/runs/art`` or ``…/a/b/c``.

    Parameters
    ----------
    value : str or pathlib.Path
        Path.
    keep : int, optional
        Trailing components kept when a long path is elided.

    Returns
    -------
    str
    """
    text = str(value)
    home = str(Path.home())
    if text.startswith(home):
        text = "~" + text[len(home) :]
    if len(text) <= 48:
        return text
    parts = Path(text).parts
    if len(parts) <= keep:
        return text
    return "…/" + "/".join(parts[-keep:])


def elapsed(seconds: float, *, precise: bool = False) -> str:
    """Human-readable duration: ``0.4s`` (precise), ``42s``, ``3m 07s``, ``1h 12m``.

    Parameters
    ----------
    seconds : float
        Duration.
    precise : bool, optional
        Show tenths of a second below one minute.

    Returns
    -------
    str
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s" if precise else f"{int(seconds)}s"
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def filesize(nbytes: float) -> str:
    """Human-readable size: ``812 B``, ``4.3 MB``, ``11.7 GB`` (decimal units).

    Parameters
    ----------
    nbytes : float
        Size in bytes.

    Returns
    -------
    str
    """
    nbytes = float(max(0, nbytes))
    if nbytes < 1000:
        return f"{nbytes:.0f} B"
    for unit in ("KB", "MB"):
        nbytes /= 1000
        if nbytes < 1000:
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes / 1000:.1f} GB"


def folder_size(folder) -> str:
    """Total size of the files under ``folder``, human-readable.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder.

    Returns
    -------
    str
    """
    return filesize(
        sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file())
    )


def resources() -> str | None:
    """``CPU 62% · RAM 18.4/32.0 GB`` when psutil is installed, else ``None``.

    Returns
    -------
    str or None
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
    except Exception:
        return None
    return f"CPU {cpu:.0f}% · RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB"
