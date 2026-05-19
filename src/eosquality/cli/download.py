"""CLI handler for ``eosquality download`` — prefetch the canonical library.

This is the **only** code path in eosquality that hits the network. It writes
the artifacts to ``~/.eosquality/indices/`` and ``~/.eosquality/libraries/``.
``fit`` and the resolve helpers (``reference_library_path`` /
``reference_library_csv_path``) are local-only — they raise
``FileNotFoundError`` if the library isn't present.

No-op if a valid cached copy already exists (use ``--force`` to redownload).
Honors ``EOSQUALITY_REFERENCE_BASE_URL`` (staging bucket override).
"""

import argparse
import sys

from eosquality import set_verbosity
from eosquality.library.download import (
    ensure_library_downloaded,
    ensure_single_file_downloaded,
)
from eosquality.library.identity import (
    LIBRARY_ID,
    library_csv_filename,
    library_csv_url,
    library_dirname,
    reference_base_url,
    user_cache_dir,
    user_library_csv_cache_dir,
)


def cmd_download(args: argparse.Namespace) -> int:
    """Argparse handler for ``eosquality download``."""
    if args.verbose:
        set_verbosity(True)
    try:
        csv_path = ensure_single_file_downloaded(
            url=library_csv_url(),
            dest=user_library_csv_cache_dir() / library_csv_filename(),
            force=args.force,
        )
        index_path = ensure_library_downloaded(
            base_url=reference_base_url(),
            dirname=library_dirname(),
            cache_dir=user_cache_dir(),
            expected_library_id=LIBRARY_ID,
            force=args.force,
        )
    except Exception as exc:
        print(f"error: could not download reference library: {exc}", file=sys.stderr)
        return 1
    print(f"Library CSV    → {csv_path}")
    print(f"Library index  → {index_path}")
    return 0


def register_subparsers(subparsers) -> None:
    """Attach the ``download`` subcommand to *subparsers*."""
    download_p = subparsers.add_parser(
        "download",
        help="Prefetch the canonical reference library into the local cache.",
        description=(
            "Explicitly download the reference library for this release of "
            "eosquality into ~/.eosquality/indices/ and ~/.eosquality/libraries/. "
            "This is the only command that hits the network — 'fit' itself is "
            "local-only and will fail with FileNotFoundError if the library "
            "isn't found in the cwd or the user cache. "
            "If a valid cached copy already exists, this is a no-op unless "
            "--force is passed. Honors EOSQUALITY_REFERENCE_BASE_URL for "
            "staging buckets."
        ),
    )
    download_p.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Redownload even if a valid cached copy already exists.",
    )
    download_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information.",
    )
    download_p.set_defaults(func=cmd_download)
