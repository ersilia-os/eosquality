"""Shared console, accent colour and the status-line primitives."""

from __future__ import annotations

import psutil
from rich.console import Console

#: The one console for the whole package (stderr keeps stdout free for data).
#: ``highlight=False``: all colour comes from explicit markup, not Rich's
#: number highlighter, which would speckle digits against the step palette.
console = Console(stderr=True, highlight=False)

# The first non-blocking CPU reading is always 0.0; take it now so the first
# section rule shows a real value.
psutil.cpu_percent(interval=None)

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


def active_color() -> str:
    """The current accent colour.

    Returns
    -------
    str
    """
    return _active_color


def set_active_color(color: str) -> None:
    """Set the accent colour used by rules, steps and panels.

    Parameters
    ----------
    color : str
        A Rich colour name.
    """
    global _active_color
    _active_color = color or "cyan"


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
