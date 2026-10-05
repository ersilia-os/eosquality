"""``eosquality download`` — prefetch the canonical reference library.

This is the **only** code path in eosquality that hits the network. It writes
the artifacts to ``~/.eosquality/indices/`` and ``~/.eosquality/libraries/``.
``fit`` and the resolve helpers are local-only. No-op if a valid cached copy
already exists (use ``--force`` to redownload). Honors
``EOSQUALITY_REFERENCE_BASE_URL`` (staging bucket override).
"""

import click

from eosquality.cli._common import CliError, run_command, say, verbose_option
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


@click.command(
    "download",
    help=(
        "Download the reference library for this eosquality release into "
        "~/.eosquality/. The only command that uses the network; a no-op if a "
        "valid cached copy exists. Honors EOSQUALITY_REFERENCE_BASE_URL."
    ),
    short_help="Prefetch the canonical reference library into the local cache.",
)
@click.option(
    "--force", "-f", is_flag=True, help="Redownload even if a valid cached copy exists."
)
@verbose_option
def download(force: bool, verbose: bool) -> None:
    """Download the reference library CSV and index into the user cache.

    Parameters
    ----------
    force : bool
        Redownload even if a valid cached copy exists.
    verbose : bool
        Print debug messages.
    """

    def work():
        try:
            csv_path = ensure_single_file_downloaded(
                url=library_csv_url(),
                dest=user_library_csv_cache_dir() / library_csv_filename(),
                force=force,
            )
            index_path = ensure_library_downloaded(
                base_url=reference_base_url(),
                dirname=library_dirname(),
                cache_dir=user_cache_dir(),
                expected_library_id=LIBRARY_ID,
                force=force,
            )
        except Exception as exc:
            raise CliError(f"could not download reference library: {exc}") from exc
        say(f"Library CSV    → {csv_path}", err=False)
        say(f"Library index  → {index_path}", err=False)

    run_command(work, verbose=verbose)
