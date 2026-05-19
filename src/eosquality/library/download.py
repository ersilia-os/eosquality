"""Fetch the canonical reference library from its public S3 URL.

The library is not shipped in the wheel — it lives in a public S3 bucket and
is downloaded lazily on first use into a user cache (``~/.eosquality/``).
The maintainer side uses ``eosvc`` to push updates to S3, but at runtime we
do not depend on ``eosvc`` — plain HTTPS is enough for public objects.

One function is public: :func:`ensure_library_downloaded`. It either returns
the local cache path if it's already present and valid, or fetches the five
known files (``vector_index.h5``, ``knn_distances.npy``, ``knn_indices.npy``,
``smiles.csv``, ``metadata.json``), writes them to a temp directory, and then
atomically moves them into the final cache location. Partial downloads never
leave a half-populated cache folder.

Progress output goes to stderr via ``rich`` (already a dep) — a per-file
progress bar with byte counts, transfer speed, and ETA. Cached hits and
error paths print compact status lines instead of a bar.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import urllib.error
import urllib.request

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Files that make up a complete reference library folder. Must stay in sync
# with what ``eosquality build`` emits (see vectorindex/backend.py::build).
_LIBRARY_FILES: tuple[str, ...] = (
    "vector_index.h5",
    "knn_distances.npy",
    "knn_indices.npy",
    "smiles.csv",
    "metadata.json",
)

# Chunk size for streamed copy. 256 KB is the sweet spot for progress
# update frequency vs syscall overhead on typical networks.
_CHUNK_BYTES = 256 * 1024

# Downloads are a user-triggered, non-trivial operation: always show progress
# on stderr regardless of the global logger verbosity. Stdout stays clean for
# machine-readable CLI output.
_console = Console(stderr=True, highlight=False)


class LibraryDownloadError(RuntimeError):
    """Raised when the reference library cannot be fetched or verified."""


def ensure_library_downloaded(
    base_url: str,
    dirname: str,
    cache_dir: pathlib.Path,
    expected_library_id: str,
    force: bool = False,
) -> pathlib.Path:
    """Return a local path to the reference library, fetching it if missing.

    Parameters
    ----------
    base_url:
        Public HTTPS prefix under which library folders live. Must end with
        ``/``. Example: ``https://eosvc-public.s3.amazonaws.com/eosquality/indices/``.
    dirname:
        Folder name on S3 and in the local cache (e.g. ``ersilia_reference_library_v0``).
    cache_dir:
        Parent directory for cached libraries. The resolved library path is
        ``cache_dir / dirname``.
    expected_library_id:
        Value that the downloaded ``metadata.json`` ``library_name`` must equal.
        Protects against a stale or wrong bucket being configured.
    force:
        If True, redownload even when a cached copy exists.
    """
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    library_dir = cache_dir / dirname

    if not force and _is_library_cached_and_valid(library_dir, expected_library_id):
        _console.print(
            f"[dim]↪ reference library cached [/dim]"
            f"[cyan]{dirname}[/cyan] [dim]→[/dim] {library_dir}"
        )
        return library_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    _console.rule(f"[bold]Downloading reference library[/bold] [cyan]{dirname}[/cyan]")
    _console.print(f"[dim]source:[/dim]      {base_url}{dirname}/")
    _console.print(f"[dim]destination:[/dim] {library_dir}")
    if force:
        _console.print("[dim]mode:[/dim]        force (redownloading)")

    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix=f".{dirname}.", dir=cache_dir) as tmp_str:
        tmp_dir = pathlib.Path(tmp_str)
        with _build_progress() as progress:
            for filename in _LIBRARY_FILES:
                src = f"{base_url}{dirname}/{filename}"
                dst = tmp_dir / filename
                total_bytes += _download_one(src, dst, progress)

        # Integrity: the library_name inside the just-fetched metadata.json
        # must match what we asked for. Mismatch means the remote was moved
        # out from under us or the base URL is misconfigured.
        fetched_id = _read_library_name(tmp_dir / "metadata.json")
        if fetched_id != expected_library_id:
            raise LibraryDownloadError(
                f"Downloaded reference library has library_name "
                f"{fetched_id!r} but this eosquality expects "
                f"{expected_library_id!r}. Wrong base URL or stale bucket."
            )
        _console.print(
            f"[green]✓[/green] integrity check passed "
            f"[dim](library_name={fetched_id!r})[/dim]"
        )

        # Atomic swap: remove any existing (stale/partial) folder, then move
        # the verified tmp folder into place. shutil.move handles cross-fs
        # case even though tmp is inside cache_dir here.
        if library_dir.exists():
            shutil.rmtree(library_dir)
        shutil.move(str(tmp_dir), str(library_dir))

    _console.print(
        f"[green]✓[/green] reference library ready "
        f"[dim]({_fmt_bytes(total_bytes)} across {len(_LIBRARY_FILES)} files)[/dim] "
        f"→ {library_dir}"
    )
    _console.rule()
    return library_dir


def _build_progress() -> Progress:
    """rich.Progress with per-file bar, bytes, speed, and ETA columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[filename]}[/bold cyan]"),
        BarColumn(bar_width=None),
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=_console,
        transient=False,
    )


def _download_one(src: str, dst: pathlib.Path, progress: Progress) -> int:
    """Stream one file to ``dst`` with a per-file progress task.

    Returns the number of bytes written. Raises LibraryDownloadError on any
    failure. Unknown Content-Length (rare for S3) produces an indeterminate
    bar — progress text still shows bytes downloaded.
    """
    try:
        response = urllib.request.urlopen(src)
    except urllib.error.HTTPError as exc:
        raise LibraryDownloadError(
            f"HTTP {exc.code} fetching {src}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LibraryDownloadError(
            f"Network error fetching {src}: {exc.reason}. "
            "Check your connection, or set EOSQUALITY_REFERENCE_LIBRARY_PATH "
            "to point at a pre-downloaded folder."
        ) from exc

    total = response.headers.get("Content-Length")
    try:
        total_int = int(total) if total is not None else None
    except ValueError:
        total_int = None

    task_id = progress.add_task(
        "download",
        total=total_int,
        filename=dst.name,
    )

    written = 0
    with response, open(dst, "wb") as out:
        while True:
            chunk = response.read(_CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            progress.update(task_id, advance=len(chunk))
    # Finalize the task so the bar reads 100% even if Content-Length was off.
    if total_int is None:
        progress.update(task_id, total=written, completed=written)
    return written


def _is_library_cached_and_valid(
    library_dir: pathlib.Path, expected_library_id: str
) -> bool:
    """Return True if every expected file is present and metadata matches."""
    if not library_dir.is_dir():
        return False
    for filename in _LIBRARY_FILES:
        if not (library_dir / filename).is_file():
            return False
    try:
        return _read_library_name(library_dir / "metadata.json") == expected_library_id
    except (OSError, ValueError):
        return False


def _read_library_name(metadata_path: pathlib.Path) -> str:
    with open(metadata_path) as f:
        data = json.load(f)
    return str(data.get("library_name", ""))


# ---------------------------------------------------------------------------
# Single-file download (used for the library SMILES CSV)
# ---------------------------------------------------------------------------


def ensure_single_file_downloaded(
    url: str,
    dest: pathlib.Path,
    force: bool = False,
) -> pathlib.Path:
    """Return a local path to a single file, downloading it if missing.

    Atomically writes to a sibling ``.part`` path first, then renames into
    ``dest``. If ``dest`` already exists and ``force`` is False, returns it
    without touching the network.
    """
    if not force and dest.is_file():
        _console.print(
            f"[dim]↪ file cached[/dim] [cyan]{dest.name}[/cyan] [dim]→[/dim] {dest}"
        )
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    _console.rule(f"[bold]Downloading[/bold] [cyan]{dest.name}[/cyan]")
    _console.print(f"[dim]source:[/dim]      {url}")
    _console.print(f"[dim]destination:[/dim] {dest}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with _build_progress() as progress:
            _download_one(url, tmp, progress)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    _console.print(f"[green]✓[/green] file ready → {dest}")
    _console.rule()
    return dest


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count (e.g. ``1.6 MB``)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"  # unreachable; keeps type-checker happy
