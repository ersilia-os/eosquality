"""``eosquality setup`` — fetch the canonical reference library into the local cache.

This is the **only** code path in eosquality that hits the network. It writes
the library folder to ``~/.eosquality/indices/``. ``fit`` and the resolve
helpers are local-only. No-op if a valid cached copy
already exists (use ``--force`` to fetch again). Honours
``EOSQUALITY_REFERENCE_BASE_URL`` (staging bucket override).
"""

import time

import click

from eosquality.cli._common import CliError, run_command, verbose_option
from eosquality.library.identity import (
    LIBRARY_ID,
    library_dirname,
    reference_base_url,
    user_cache_dir,
)
from eosquality.utils import console


@click.command(
    "setup",
    help=(
        "Set up eosquality: fetch the reference library for this release into "
        "~/.eosquality/. The only command that uses the network; a no-op if a "
        "valid cached copy exists. Honours EOSQUALITY_REFERENCE_BASE_URL."
    ),
    short_help="Fetch the canonical reference library into the local cache.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Fetch again even if a valid cached copy exists.",
)
@verbose_option
def setup(force: bool, verbose: bool) -> None:
    """Fetch the reference library folder into the user cache.

    Parameters
    ----------
    force : bool
        Fetch again even if a valid cached copy exists.
    verbose : bool
        Print debug messages.
    """

    def _work():
        started = time.perf_counter()
        console.summary_panel(
            "eosquality · setup",
            [
                ("library", LIBRARY_ID),
                ("cache", console.path(user_cache_dir())),
                ("source", console.plain(reference_base_url())),
            ],
            icon="◆",
        )
        from eosquality.library.download import ensure_library_downloaded

        with console.section("Reference library") as section:
            try:
                library = ensure_library_downloaded(
                    base_url=reference_base_url(),
                    dirname=library_dirname(),
                    cache_dir=user_cache_dir(),
                    expected_library_id=LIBRARY_ID,
                    force=force,
                )
            except Exception as exc:
                raise CliError(f"could not fetch the reference library: {exc}") from exc
            section.summary = "ready"
        console.summary_panel(
            "Setup complete",
            [
                ("library folder", console.path(library)),
                ("time", console.elapsed(time.perf_counter() - started)),
            ],
            color="green",
            icon="✓",
        )

    run_command(_work, verbose=verbose, command="setup")
