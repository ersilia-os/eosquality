"""Top-level command-line interface for eosquality.

Each subcommand lives in its own module under this package:

- :mod:`eosquality.cli.build` defines ``build``.
- :mod:`eosquality.cli.download` defines ``download``.
- :mod:`eosquality.cli.fit` defines ``fit``.
- :mod:`eosquality.cli.run` defines ``run``.

This module is the Click group that dispatches to them.

End-user workflow
-----------------
Each release is pinned to a canonical reference library, resolved locally
from ``$EOSQUALITY_REFERENCE_LIBRARY_PATH`` → ``./data/indices/<library>/``
→ ``~/.eosquality/indices/<library>/``. Fetch it once with
``eosquality download``; after that the usual path is fit then run::

    eosquality fit --reference eos4e40_v1.csv --output artifacts/ [--training training_eos4e40_v1/]
    eosquality run --input query.csv --artifacts artifacts/ --output scores.csv

Prefetch the library explicitly (useful for CI or airgapped setups)::

    eosquality download [--force]

For maintainers / advanced use
------------------------------
``eosquality build`` rebuilds the vector index from a SMILES library CSV.
It is a release tool — ordinary users should not need to run it. Use it
to produce a new canonical library for the next major release, or to
build a non-canonical index for internal testing and fit against it::

    eosquality build --input library.csv --output /tmp/idx/ [--max-k 50]
    eosquality fit --reference eos4e40_v1.csv --output artifacts/ --vector-index /tmp/idx/
"""

import importlib.metadata

import click

from eosquality import set_log_level
from eosquality.cli.build import build
from eosquality.cli.download import download
from eosquality.cli.fit import fit
from eosquality.cli.run import run

try:
    _VERSION = importlib.metadata.version("eosquality")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"


@click.group(
    help="Assess the quality of Ersilia model predictions.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(_VERSION, prog_name="eosquality")
def cli() -> None:
    """Show INFO-level progress for every subcommand (``-v`` adds DEBUG)."""
    set_log_level("INFO")


cli.add_command(build)
cli.add_command(download)
cli.add_command(fit)
cli.add_command(run)


def main(argv: list[str] | None = None) -> None:
    """Run the CLI on ``argv`` (default ``sys.argv[1:]``) and exit with its status.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments, without the program name.
    """
    cli.main(args=argv, prog_name="eosquality")


if __name__ == "__main__":
    main()
