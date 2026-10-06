"""Progress bars for long per-molecule loops, on the shared curated console.

Kept as a thin alias of :func:`eosquality.utils.console.progress` so every
bar is themed with the active command colour and stays silent while the
curated output is off (library use).
"""

from __future__ import annotations

from rich.progress import Progress

from eosquality.utils import console


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
    return console.progress(label)
