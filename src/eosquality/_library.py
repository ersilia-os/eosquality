"""Canonical reference library pinned to this package release.

The library is **not** bundled in the wheel. It lives in a public S3 bucket
and is downloaded lazily into
``~/.eosquality/indices/ersilia_reference_library_vN/`` on first use. The
maintainer pushes new libraries to S3 with ``eosvc``; the runtime side only
needs plain HTTPS.

One canonical name, ``ersilia_reference_library_vN``, is used everywhere:
the content identity written into each library's ``metadata.json``
``library_name`` field (:data:`LIBRARY_ID`), the source CSV stem under
``data/libraries/``, the folder under ``data/indices/``, the local cache
folder, and the S3 path segment.

A new reference library — adding or removing molecules, rebuilding the
vector index with different Morgan parameters, or correcting SMILES in
place — changes scores and therefore requires a major version bump of
the package. Metadata-only edits (description, citation) do not bump.

Environment variables:

- ``EOSQUALITY_REFERENCE_BASE_URL``: override the S3 base URL. Useful for
  staging a test bucket or for CI. Must end with ``/``.
- ``EOSQUALITY_REFERENCE_LIBRARY_PATH``: point at a pre-placed folder
  (e.g. a dev checkout's ``data/indices/ersilia_reference_library_v0/``)
  and skip download entirely. Lets you work offline and lets CI avoid the
  network.
"""

from __future__ import annotations

import os
import pathlib
import re

from rich.console import Console

from eosquality._download import (
    _is_library_cached_and_valid,
    ensure_library_downloaded,
    ensure_single_file_downloaded,
)

_console = Console(stderr=True, highlight=False)

LIBRARY_ID: str = "ersilia_reference_library_v0"

_LIBRARY_ID_RE = re.compile(r"^ersilia_reference_library_v(\d+)$")

# Public S3 prefix for published reference libraries. The maintainer pushes
# ``data/indices/ersilia_reference_library_vN/`` folders here via eosvc;
# clients pull via plain HTTPS.
DEFAULT_REFERENCE_BASE_URL: str = (
    "https://eosvc-public.s3.amazonaws.com/eosquality/indices/"
)


def library_major() -> int:
    """Return the major version number encoded in :data:`LIBRARY_ID`."""
    match = _LIBRARY_ID_RE.match(LIBRARY_ID)
    if match is None:
        raise RuntimeError(
            f"LIBRARY_ID {LIBRARY_ID!r} does not match "
            "'ersilia_reference_library_v<N>' — release-engineering bug."
        )
    return int(match.group(1))


def library_dirname() -> str:
    """Folder name used on S3, in the repo's ``data/indices/``, and in the cache.

    Equal to :data:`LIBRARY_ID` — a single canonical name lines up everywhere:
    source CSV stem, ``metadata.json`` ``library_name``, local folder, cache
    path, and S3 URL segment.
    """
    return LIBRARY_ID


def reference_base_url() -> str:
    """Effective base URL (env override or baked-in default), guaranteed to end with ``/``."""
    url = os.environ.get("EOSQUALITY_REFERENCE_BASE_URL", DEFAULT_REFERENCE_BASE_URL)
    return url if url.endswith("/") else url + "/"


def user_cache_dir() -> pathlib.Path:
    """Root of the local cache for downloaded library *indices* (``~/.eosquality/indices/``)."""
    return pathlib.Path.home() / ".eosquality" / "indices"


def user_library_csv_cache_dir() -> pathlib.Path:
    """Root of the local cache for downloaded library *source CSVs*
    (``~/.eosquality/libraries/``)."""
    return pathlib.Path.home() / ".eosquality" / "libraries"


def library_csv_filename() -> str:
    """Filename for the canonical library SMILES CSV (``<LIBRARY_ID>.csv``)."""
    return f"{LIBRARY_ID}.csv"


def library_csv_url() -> str:
    """Public HTTPS URL for the canonical library CSV.

    Mirrors :func:`reference_base_url` but under the ``libraries/`` prefix —
    S3 path layout is::

        <bucket>/eosquality/indices/<LIBRARY_ID>/ (index folder)
        <bucket>/eosquality/libraries/<LIBRARY_ID>.csv (source SMILES)

    The base URL is shared between the two via env override
    ``EOSQUALITY_REFERENCE_BASE_URL``; we swap the trailing ``indices/`` for
    ``libraries/``.
    """
    indices_url = reference_base_url()
    # Replace trailing "indices/" with "libraries/" to find the sibling prefix.
    if indices_url.endswith("/indices/"):
        libraries_url = indices_url[: -len("indices/")] + "libraries/"
    else:
        libraries_url = indices_url.rstrip("/") + "/../libraries/"
    return libraries_url + library_csv_filename()


def _cwd_library_candidate() -> pathlib.Path:
    """Where to look in the current working directory for a pre-placed library.

    Matches the repo convention: running from the eosquality checkout (or any
    workdir that mirrors it) should Just Work without env vars.
    """
    return pathlib.Path.cwd() / "data" / "indices" / library_dirname()


def _cwd_library_csv_candidate() -> pathlib.Path:
    """Where to look in the current working directory for a pre-placed library CSV."""
    return pathlib.Path.cwd() / "data" / "libraries" / library_csv_filename()


def reference_library_path(force_download: bool = False) -> pathlib.Path:
    """Return a local filesystem path to the reference library.

    Resolution order:

    1. If ``EOSQUALITY_REFERENCE_LIBRARY_PATH`` is set, use it verbatim — no
       download, no cache check. Intended for dev/CI: point at your
       ``data/indices/ersilia_reference_library_vN/`` checkout, or at any
       folder already verified offline.
    2. Otherwise, check ``./data/indices/<library_dirname>/`` relative to the
       current working directory. If present and valid (files + metadata
       match), use it — zero network, zero configuration when running from
       a repo checkout or a matching workdir layout.
    3. Otherwise, check the user cache (``~/.eosquality/indices/...``). If
       complete and valid, return it.
    4. Otherwise, download from :func:`reference_base_url` into the cache
       (atomically) and return the cached path.

    Raises
    ------
    FileNotFoundError
        If the env-override path is set but doesn't exist.
    eosquality._download.LibraryDownloadError
        If the download or integrity check fails.
    """
    override = os.environ.get("EOSQUALITY_REFERENCE_LIBRARY_PATH")
    if override:
        path = pathlib.Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"EOSQUALITY_REFERENCE_LIBRARY_PATH={override!r} does not exist."
            )
        return path

    cwd_candidate = _cwd_library_candidate()
    if not force_download and _is_library_cached_and_valid(cwd_candidate, LIBRARY_ID):
        _console.print(
            f"[dim]↪ reference library found in cwd[/dim] "
            f"[cyan]{library_dirname()}[/cyan] [dim]→[/dim] {cwd_candidate}"
        )
        return cwd_candidate

    return ensure_library_downloaded(
        base_url=reference_base_url(),
        dirname=library_dirname(),
        cache_dir=user_cache_dir(),
        expected_library_id=LIBRARY_ID,
        force=force_download,
    )


def reference_library_csv_path(force_download: bool = False) -> pathlib.Path:
    """Return a local filesystem path to the canonical library CSV.

    Mirrors :func:`reference_library_path` but for the single source CSV file.
    Resolution order:

    1. ``EOSQUALITY_REFERENCE_LIBRARY_CSV_PATH`` env var — explicit override.
    2. ``./data/libraries/<LIBRARY_ID>.csv`` in the current working directory.
    3. ``~/.eosquality/libraries/<LIBRARY_ID>.csv`` (user cache).
    4. Download from :func:`library_csv_url` into the user cache.

    Unlike the index, there's no multi-file integrity check — we only verify
    that the file exists. Callers should pd.read_csv on the returned path.
    """
    override = os.environ.get("EOSQUALITY_REFERENCE_LIBRARY_CSV_PATH")
    if override:
        path = pathlib.Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"EOSQUALITY_REFERENCE_LIBRARY_CSV_PATH={override!r} does not exist."
            )
        return path

    cwd_candidate = _cwd_library_csv_candidate()
    if not force_download and cwd_candidate.is_file():
        _console.print(
            f"[dim]↪ library CSV found in cwd[/dim] "
            f"[cyan]{library_csv_filename()}[/cyan] [dim]→[/dim] {cwd_candidate}"
        )
        return cwd_candidate

    return ensure_single_file_downloaded(
        url=library_csv_url(),
        dest=user_library_csv_cache_dir() / library_csv_filename(),
        force=force_download,
    )
