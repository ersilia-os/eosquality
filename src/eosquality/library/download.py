"""Fetch the canonical reference library from its public S3 URL.

The library is not shipped in the wheel: ``eosquality setup`` fetches it into a
user cache (``~/.eosquality/``). The maintainer side uses ``eosvc`` to push
updates to S3; at runtime plain HTTPS is enough for public objects.
:func:`ensure_library_downloaded` fetches every file in ``_LIBRARY_FILES`` into
a temporary folder, checks the library identity, and only then moves the
folder into the cache, so a failed download never leaves a half-populated one.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import urllib.error
import urllib.request

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from eosquality.utils import console

# Files that make up a complete reference library folder. Must stay in sync
# with what ``eosquality build`` emits: the library SMILES, its identity and
# the connectivity keys that ``ref_match`` / ``ref_scaffold`` look up.
_LIBRARY_FILES: tuple[str, ...] = (
    "smiles.csv",
    "metadata.json",
    "connectivity_keys.npz",
)

_CHUNK_BYTES = 256 * 1024
# Per-socket-operation timeout; a stalled connection fails instead of hanging.
_TIMEOUT_SECONDS = 60


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
    base_url : str
        Public HTTPS prefix under which library folders live, e.g.
        ``https://eosvc-public.s3.amazonaws.com/eosquality/indices/``.
    dirname : str
        Folder name on S3 and in the cache, e.g. ``ersilia_reference_library_v0``.
    cache_dir : pathlib.Path
        Parent directory of cached libraries.
    expected_library_id : str
        Required ``library_name`` of the downloaded ``metadata.json``.
    force : bool, optional
        Redownload even when a valid cached copy exists.

    Returns
    -------
    pathlib.Path
        ``cache_dir / dirname``.

    Raises
    ------
    LibraryDownloadError
        On a network failure or when the library identity does not match.
    """
    base_url = base_url if base_url.endswith("/") else base_url + "/"
    library_dir = cache_dir / dirname
    if not force and is_library_cached_and_valid(library_dir, expected_library_id):
        console.echo(f"library cached → {console.path(library_dir)}", "info")
        return library_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.echo(f"fetching {console.plain(base_url + dirname)}/", "info")
    with tempfile.TemporaryDirectory(prefix=f".{dirname}.", dir=cache_dir) as tmp:
        tmp_dir = pathlib.Path(tmp)
        total = _fetch_verified(base_url + dirname, tmp_dir, expected_library_id)
        if library_dir.exists():  # a stale or partial folder
            shutil.rmtree(library_dir)
        shutil.move(str(tmp_dir), str(library_dir))
    console.echo(f"{console.filesize(total)} → {console.path(library_dir)}", "success")
    return library_dir


def _fetch_verified(url: str, tmp_dir: pathlib.Path, expected_library_id: str) -> int:
    """Download every library file into ``tmp_dir``; check the library identity."""
    total = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[filename]}[/]"),
        BarColumn(bar_width=None),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console.console,
        transient=True,
    ) as progress:
        for filename in _LIBRARY_FILES:
            total += _download_one(f"{url}/{filename}", tmp_dir / filename, progress)
    fetched_id = _read_library_name(tmp_dir / "metadata.json")
    if fetched_id != expected_library_id:
        raise LibraryDownloadError(
            f"Downloaded reference library has library_name {fetched_id!r} but "
            f"this eosquality expects {expected_library_id!r}. Wrong base URL or "
            "stale bucket."
        )
    return total


def _download_one(src: str, dst: pathlib.Path, progress: Progress) -> int:
    """Stream one file to ``dst``; return the bytes written."""
    try:
        response = urllib.request.urlopen(src, timeout=_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        raise LibraryDownloadError(
            f"HTTP {exc.code} fetching {src}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LibraryDownloadError(
            f"Network error fetching {src}: {exc.reason}. Check your connection, "
            "or set EOSQUALITY_REFERENCE_LIBRARY_PATH to a pre-downloaded folder."
        ) from exc
    length = response.headers.get("Content-Length")
    task = progress.add_task(
        "download",
        total=int(length) if length and length.isdigit() else None,
        filename=dst.name,
    )
    written = 0
    with response, open(dst, "wb") as out:
        while chunk := response.read(_CHUNK_BYTES):
            out.write(chunk)
            written += len(chunk)
            progress.update(task, advance=len(chunk))
    return written


def is_library_cached_and_valid(
    library_dir: pathlib.Path, expected_library_id: str
) -> bool:
    """Whether every expected file is present and the library identity matches.

    Parameters
    ----------
    library_dir : pathlib.Path
        Candidate library folder.
    expected_library_id : str
        Required ``library_name`` in its ``metadata.json``.

    Returns
    -------
    bool
    """
    if not all((library_dir / name).is_file() for name in _LIBRARY_FILES):
        return False
    try:
        return _read_library_name(library_dir / "metadata.json") == expected_library_id
    except (OSError, ValueError):
        return False


def _read_library_name(metadata_path: pathlib.Path) -> str:
    with open(metadata_path) as f:
        return str(json.load(f).get("library_name", ""))
