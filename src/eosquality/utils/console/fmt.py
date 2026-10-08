"""Formatting helpers: paths, durations, sizes, resource use."""

from __future__ import annotations

from pathlib import Path

import psutil
from rich.markup import escape


def plural(n: int, word: str) -> str:
    """``n`` and ``word``, pluralised with an ``s`` unless ``n`` is 1.

    Parameters
    ----------
    n : int
        The count.
    word : str
        The singular noun.

    Returns
    -------
    str
        E.g. ``"1 column"``, ``"3 columns"``.
    """
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def path(value, keep: int = 3) -> str:
    """Render a path compactly (``~/runs/art`` or ``…/a/b/c``), markup-escaped.

    The result is safe to interpolate into Rich markup: a path containing
    ``[`` is shown verbatim rather than parsed as a style tag.

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
    if len(text) > 48:
        parts = Path(text).parts
        if len(parts) > keep:
            text = "…/" + "/".join(parts[-keep:])
    return escape(text)


def plain(value) -> str:
    """``str(value)`` escaped for Rich markup (user data inside a markup line).

    Parameters
    ----------
    value : object
        Anything printable.

    Returns
    -------
    str
    """
    return escape(str(value))


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


def resources() -> str:
    """``CPU 62% · RAM 18.4/32.0 GB``: system-wide use, shown on section rules.

    Returns
    -------
    str
    """
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    return f"CPU {cpu:.0f}% · RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB"
