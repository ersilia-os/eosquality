"""Canonical reference library pinned to this package release.

The library is **not** bundled in the wheel and is **not** auto-downloaded.
Resolution is local-only: env var override → repo ``data/indices/`` →
``~/.eosquality/indices/`` user cache. If none of these exist,
:func:`reference_library_path` raises ``FileNotFoundError`` and asks the user
to run ``eosquality setup`` (which writes to the user cache). The
maintainer pushes new libraries to S3 with ``eosvc``; the runtime side only
hits the network when the user explicitly invokes ``eosquality setup``.

One canonical name, ``ersilia_reference_library_vN``, is used everywhere:
the content identity written into each library's ``metadata.json``
``library_name`` field (:data:`LIBRARY_ID`), the folder under
``data/indices/``, the local cache folder, and the S3 path segment.

A new reference library — adding or removing molecules, rebuilding the
connectivity keys, or correcting SMILES in
place — changes scores and therefore requires a major version bump of
the package. Metadata-only edits (description, citation) do not bump.

Environment variables:

- ``EOSQUALITY_REFERENCE_BASE_URL``: override the S3 base URL used by
  ``eosquality setup``. Useful for staging a test bucket or for CI.
  Must end with ``/``.
- ``EOSQUALITY_REFERENCE_LIBRARY_PATH``: point at a pre-placed folder
  (e.g. a dev checkout's ``data/indices/ersilia_reference_library_v0/``)
  and short-circuit the local lookup. Lets you work entirely offline.
"""

from __future__ import annotations

import os
import pathlib
import re

LIBRARY_ID: str = "ersilia_reference_library_v0"

_LIBRARY_ID_RE = re.compile(r"^ersilia_reference_library_v(\d+)$")

# Public S3 prefix for published reference libraries. The maintainer pushes
# ``data/indices/ersilia_reference_library_vN/`` folders here via eosvc;
# clients pull via plain HTTPS.
DEFAULT_REFERENCE_BASE_URL: str = (
    "https://eosvc-public.s3.amazonaws.com/eosquality/data/indices/"
)


def is_library_id(name: str) -> bool:
    """Whether ``name`` has the form of a library id, ``ersilia_reference_library_vN``.

    Parameters
    ----------
    name : str
        A candidate library name.

    Returns
    -------
    bool
    """
    return _LIBRARY_ID_RE.match(name) is not None


def library_major() -> int:
    """Return the major version number encoded in :data:`LIBRARY_ID`.

    Returns
    -------
    int
    """
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

    Returns
    -------
    str
    """
    return LIBRARY_ID


def reference_base_url() -> str:
    """Effective base URL (env override or baked-in default), guaranteed to end with ``/``.

    Returns
    -------
    str
    """
    url = os.environ.get("EOSQUALITY_REFERENCE_BASE_URL", DEFAULT_REFERENCE_BASE_URL)
    return url if url.endswith("/") else url + "/"


def user_cache_dir() -> pathlib.Path:
    """Root of the local cache for downloaded library *indices* (``~/.eosquality/indices/``).

    Returns
    -------
    pathlib.Path
    """
    return pathlib.Path.home() / ".eosquality" / "indices"


def _cwd_library_candidate() -> pathlib.Path:
    """Where to look in the current working directory for a pre-placed library.

    Matches the repo convention: running from the eosquality checkout (or any
    workdir that mirrors it) should Just Work without env vars.
    """
    return pathlib.Path.cwd() / "data" / "indices" / library_dirname()


def reference_library_path() -> pathlib.Path:
    """Return a local filesystem path to the reference library.

    Resolution order — all **local-only**, no network:

    1. ``EOSQUALITY_REFERENCE_LIBRARY_PATH`` env var, used verbatim.
    2. ``./data/indices/<library_dirname>/`` relative to the current working
       directory, if present and valid (files + metadata match).
    3. ``~/.eosquality/indices/<library_dirname>/`` user cache, if present
       and valid (populated by ``eosquality setup``).

    Raises
    ------
    FileNotFoundError
        If none of the above are present. The error message points the user
        at the explicit ``eosquality setup`` command (or the env var, or
        the repo's ``data/indices/`` layout) — fit-time resolution never
        downloads automatically.

    Returns
    -------
    pathlib.Path
        Folder of the canonical reference library.
    """
    from eosquality.library.download import is_library_cached_and_valid
    from eosquality.utils.logging import logger

    override = os.environ.get("EOSQUALITY_REFERENCE_LIBRARY_PATH")
    if override:
        path = pathlib.Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"EOSQUALITY_REFERENCE_LIBRARY_PATH={override!r} does not exist."
            )
        return path

    cwd_candidate = _cwd_library_candidate()
    if is_library_cached_and_valid(cwd_candidate, LIBRARY_ID):
        logger.debug(f"reference library found in cwd → {cwd_candidate}")
        return cwd_candidate

    cache_candidate = user_cache_dir() / library_dirname()
    if is_library_cached_and_valid(cache_candidate, LIBRARY_ID):
        logger.debug(f"reference library found in user cache → {cache_candidate}")
        return cache_candidate

    raise FileNotFoundError(
        f"Reference library {library_dirname()!r} not found locally. "
        f"Looked at:\n"
        f"  • $EOSQUALITY_REFERENCE_LIBRARY_PATH (unset)\n"
        f"  • {cwd_candidate} (cwd)\n"
        f"  • {cache_candidate} (user cache)\n"
        f"Run 'eosquality setup' to fetch it, or set "
        f"EOSQUALITY_REFERENCE_LIBRARY_PATH to a local checkout."
    )
