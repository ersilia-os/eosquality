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

    ``fn`` must be a module-level (picklable) function. Inputs of at least
    :data:`PARALLEL_MIN_ITEMS` use ``multiprocessing.Pool.imap`` with
    ``n_jobs`` workers (default: every CPU). ``show_progress=None`` shows a
    progress bar only for parallel (library-build-sized) inputs.
    """
    n = len(items)
    if n == 0:
        return out
    n_jobs = max(1, min(n_jobs or os.cpu_count() or 1, n))
    parallel = n_jobs > 1 and n >= PARALLEL_MIN_ITEMS
    if show_progress is None:
        show_progress = parallel
    progress = make_progress(label) if show_progress else None
    task_id = progress.add_task(label, total=n) if progress is not None else None
    if progress is not None:
        progress.start()
    try:
        if parallel:
            with mp.Pool(processes=n_jobs) as pool:
                rows = pool.imap(fn, items, chunksize=chunksize)
                for i, row in enumerate(rows):
                    out[i] = row
                    if progress is not None:
                        progress.advance(task_id)
        else:
            for i, item in enumerate(items):
                out[i] = fn(item)
                if progress is not None:
                    progress.advance(task_id)
    finally:
        if progress is not None:
            progress.stop()
    return out
