"""Order-preserving per-molecule map: in-process, or a process pool on request."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager

import numpy as np

from eosquality.utils import console

# A process pool is used only when the caller asks for it (``n_jobs > 1``, or
# ``workers(n)`` around a command) and the input is large enough: spawning
# workers (each re-imports RDKit) costs more than it saves on small inputs.
# Library code never asks by default: with the "spawn" start method (macOS,
# Windows) a pool started from a user's script without an
# ``if __name__ == "__main__":`` guard re-executes that script in every worker.
# The CLI, a proper entry point, opts in through :func:`workers`.
PARALLEL_MIN_ITEMS = 5_000
# ``-1`` means every core up to this many: each worker imports RDKit (~200 MB),
# and past a handful of workers the spawn cost eats the gain.
MAX_AUTO_WORKERS = 16
_default_jobs: int | None = None


@contextmanager
def workers(n_jobs: int | None):
    """Let :func:`map_rows` calls inside the block use ``n_jobs`` processes.

    Parameters
    ----------
    n_jobs : int or None
        Worker processes (``-1``: every CPU; ``None`` or 1: none).
    """
    global _default_jobs
    previous, _default_jobs = _default_jobs, n_jobs
    try:
        yield
    finally:
        _default_jobs = previous


def map_rows(
    fn: Callable[[str], np.ndarray],
    items: Sequence[str],
    out: np.ndarray,
    *,
    label: str,
    n_jobs: int | None = None,
    chunksize: int = 256,
    min_items: int = PARALLEL_MIN_ITEMS,
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
        Worker processes. ``None`` takes the :func:`workers` setting (in-process
        outside one); 1 runs in-process; ``-1`` uses every CPU, up to
        :data:`MAX_AUTO_WORKERS`.
    chunksize : int, optional
        Items per task sent to a worker.
    min_items : int, optional
        Fewest items for which a pool is worth starting.
    show_progress : bool, optional
        Show a progress bar; ``None`` shows it only for parallel runs.

    Returns
    -------
    numpy.ndarray
        ``out``, filled. A process pool is used only with ``n_jobs`` other
        than 1 and at least ``min_items`` inputs.
    """
    n = len(items)
    if n == 0:
        return out
    if n_jobs is None:
        n_jobs = _default_jobs
    if n_jobs is not None and n_jobs < 0:
        n_jobs = min(os.cpu_count() or 1, MAX_AUTO_WORKERS)
    n_jobs = max(1, min(n_jobs or 1, n))
    parallel = n_jobs > 1 and n >= min_items
    pool = None
    if parallel:
        # Before the progress bar's thread exists: a forked worker must not
        # inherit a lock that thread holds.
        with _single_threaded_workers():
            pool = mp.Pool(processes=n_jobs)  # workers inherit the environment
    progress = (
        console.progress(label)
        if (show_progress is None and parallel) or show_progress
        else None
    )
    task_id = progress.add_task(label, total=n) if progress is not None else None
    if progress is not None:
        progress.start()
    try:
        if pool is not None:
            with pool:
                _fill(out, pool.imap(fn, items, chunksize=chunksize), progress, task_id)
        else:
            _fill(out, map(fn, items), progress, task_id)
    finally:
        if progress is not None:
            progress.stop()
    return out


@contextmanager
def _single_threaded_workers():
    """Make workers started inside the block inherit one BLAS thread each.

    RDKit descriptors call into numpy linear algebra; with every worker also
    running a multi-threaded BLAS the cores are oversubscribed and the pool
    ends up slower than a single process. The variables are restored on exit.
    """
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names if previous[name] is None})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)


def _fill(out: np.ndarray, rows, progress, task_id) -> None:
    for i, row in enumerate(rows):
        out[i] = row
        if progress is not None:
            progress.advance(task_id)
