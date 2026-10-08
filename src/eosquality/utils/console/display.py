"""Detail blocks, tables and summary panels."""

from __future__ import annotations

from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

from eosquality.utils.console._core import active_color, console, enabled
from eosquality.utils.console.fmt import plain


def median_summary(values) -> str:
    """``"median 0.421"``, or ``"no scored molecule"`` when all values are NaN.

    Parameters
    ----------
    values : pandas.Series or numpy.ndarray
        A score column, possibly all NaN (every query SMILES unparsable).

    Returns
    -------
    str
    """
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return "no scored molecule"
    return f"median {float(np.median(finite)):.3f}"


def share_summary(flags) -> str:
    """``"31% of 1,000"``: the share of ones among the known values of a flag column.

    Parameters
    ----------
    flags : pandas.Series
        A 1 / 0 column, with NA where the question has no answer.

    Returns
    -------
    str
        ``"no valid rows"`` when every value is missing.
    """
    known = flags.dropna()
    return f"{known.mean():.0%} of {len(known):,}" if len(known) else "no valid rows"


def detail(rows, *, indent: int = 6) -> None:
    """Borderless block of dim right-aligned labels and values (shown verbatim).

    Parameters
    ----------
    rows : iterable of (str, object)
        Label/value pairs.
    indent : int, optional
        Left padding.
    """
    if not enabled():
        return
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(justify="left", overflow="fold")
    for key, value in rows:
        table.add_row(plain(key), plain(value))
    console.print(Padding(table, (0, 0, 0, indent)))


# Rows a table prints before it is cut short; the rest stay in the log file.
MAX_TABLE_ROWS = 15


def table(
    columns,
    rows,
    *,
    title: str | None = None,
    indent: int = 6,
    max_rows: int | None = MAX_TABLE_ROWS,
) -> None:
    """Borderless table with a themed header row; cells are shown verbatim.

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
    max_rows : int, optional
        Print at most this many rows, then a ``… and N more`` line (the full
        table is in the log). ``None`` prints every row.
    """
    if not enabled():
        return
    rows = list(rows)
    hidden = 0
    if max_rows is not None and len(rows) > max_rows:
        hidden = len(rows) - max_rows
        rows = rows[:max_rows]
    out = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style=f"bold {active_color()}",
        title=f"[dim]{title}[/]" if title else None,
        title_justify="left",
    )
    for i, name in enumerate(columns):
        out.add_column(name, justify="left" if i == 0 else "right", no_wrap=i == 0)
    for row in rows:
        out.add_row(*(plain(v) for v in row))
    if hidden:
        out.add_row(f"… and {hidden:,} more", *[""] * (len(columns) - 1))
    console.print(Padding(out, (0, 0, 0, indent)))


def summary_panel(
    title: str, rows, *, color: str | None = None, icon: str = ""
) -> None:
    """Rounded, left-titled panel of key/value rows (run header, final summary).

    Values are Rich markup: wrap user data in :func:`path` or :func:`plain`.

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
    if not enabled():
        return
    color = color or active_color()
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
