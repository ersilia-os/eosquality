"""Order-preserving per-molecule map: serial for small inputs, a process pool for large ones."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Sequence

import numpy as np

from eosquality.utils.progress import make_progress

# Below this many items the work runs in-process: spawning a pool (each
# worker re-imports RDKit) costs more than it saves, and run-time callers
# never need an ``if __name__ == "__main__":`` guard in their scripts.
PARALLEL_MIN_ITEMS = 5_000


def map_rows(
    fn: Callable[[str], np.ndarray],
    items: Sequence[str],
    out: np.ndarray,
    *,
    label: str,
    n_jobs: int | None = None,
    chunksize: int = 256,
    show_progress: bool | None = None,
) -> np.ndarray:
    """Fill ``out[i] = fn(items[i])`` for every item and return ``out``.

    Parameters
    ----------
    fn : callable
        Module-level (picklable) function mapping one item to one row.
    items : sequence of str
        Inputs, one per output row.
    out : numpy.ndarray
        Preallocated output with ``len(items)`` rows.
    label : str
        Progress-bar title.
    n_jobs : int, optional
        Worker processes for parallel runs (default: every CPU).
    chunksize : int, optional
        Items per task sent to a worker.
    show_progress : bool, optional
        Show a progress bar; ``None`` shows it only for parallel runs.

    Returns
    -------
    numpy.ndarray
        ``out``, filled. Inputs of at least :data:`PARALLEL_MIN_ITEMS` use a
        process pool; smaller ones run in-process.
    """
    n = len(items)
    if n == 0:
        return out
    n_jobs = max(1, min(n_jobs or os.cpu_count() or 1, n))
    parallel = n_jobs > 1 and n >= PARALLEL_MIN_ITEMS
    progress = (
        make_progress(label)
        if (show_progress is None and parallel) or show_progress
        else None
    )
    task_id = progress.add_task(label, total=n) if progress is not None else None
    if progress is not None:
        progress.start()
    try:
        if parallel:
            with mp.Pool(processes=n_jobs) as pool:
                _fill(out, pool.imap(fn, items, chunksize=chunksize), progress, task_id)
        else:
            _fill(out, map(fn, items), progress, task_id)
    finally:
        if progress is not None:
            progress.stop()
    return out


def _fill(out: np.ndarray, rows, progress, task_id) -> None:
    for i, row in enumerate(rows):
        out[i] = row
        if progress is not None:
            progress.advance(task_id)
