"""``eosquality setup`` — fetch the canonical reference library into the local cache.

This is the **only** code path in eosquality that hits the network. It writes
the artifacts to ``~/.eosquality/indices/`` and ``~/.eosquality/libraries/``.
``fit`` and the resolve helpers are local-only. No-op if a valid cached copy
already exists (use ``--force`` to fetch again). Honors
``EOSQUALITY_REFERENCE_BASE_URL`` (staging bucket override).
"""

import time

import click

from eosquality.cli._common import CliError, run_command, verbose_option
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
from eosquality.utils import console


@click.command(
    "setup",
    help=(
        "Set up eosquality: fetch the reference library for this release into "
        "~/.eosquality/. The only command that uses the network; a no-op if a "
        "valid cached copy exists. Honors EOSQUALITY_REFERENCE_BASE_URL."
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
    """Fetch the reference library CSV and index into the user cache.

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
                ("source", reference_base_url()),
            ],
            icon="◆",
        )
        steps = console.Steps(2)
        with console.section("Reference library") as section:
            try:
                with steps("Library SMILES (CSV)") as st:
                    csv_path = ensure_single_file_downloaded(
                        url=library_csv_url(),
                        dest=user_library_csv_cache_dir() / library_csv_filename(),
                        force=force,
                    )
                    st.summary = console.path(csv_path)
                with steps("Vector index and descriptors") as st:
                    index_path = ensure_library_downloaded(
                        base_url=reference_base_url(),
                        dirname=library_dirname(),
                        cache_dir=user_cache_dir(),
                        expected_library_id=LIBRARY_ID,
                        force=force,
                    )
                    st.summary = console.path(index_path)
            except Exception as exc:
                raise CliError(f"could not fetch the reference library: {exc}") from exc
            section.summary = "ready"
        console.summary_panel(
            "Setup complete",
            [
                ("library CSV", console.path(csv_path)),
                ("library index", console.path(index_path)),
                ("time", console.elapsed(time.perf_counter() - started)),
            ],
            color="green",
            icon="✓",
        )

    run_command(_work, verbose=verbose, command="setup")
