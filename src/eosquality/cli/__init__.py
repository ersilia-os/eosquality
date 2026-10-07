"""Top-level command-line interface for eosquality.

Each subcommand lives in its own module under this package:

- :mod:`eosquality.cli.build` defines ``build``.
- :mod:`eosquality.cli.setup` defines ``setup``.
- :mod:`eosquality.cli.fit` defines ``fit``.
- :mod:`eosquality.cli.run` defines ``run``.

This module is the Click group that dispatches to them.

End-user workflow
-----------------
Each release is pinned to a canonical reference library, resolved locally
from ``$EOSQUALITY_REFERENCE_LIBRARY_PATH`` → ``./data/indices/<library>/``
→ ``~/.eosquality/indices/<library>/``. Fetch it once with
``eosquality setup``; after that the usual path is fit then run::

    eosquality fit -r reference_eos4e40_v1.csv -a artifacts_eos4e40_v1/ [-t training_eos4e40_v1/]
    eosquality run -i query_eos4e40_v1.csv -a artifacts_eos4e40_v1/ -o quality_eos4e40_v1.csv

File and folder names carry the model: ``[prefix_]<eos_id>_<version>``.

Fetch the library explicitly (useful for CI or airgapped setups)::

    eosquality setup [--force]

For maintainers / advanced use
------------------------------
``eosquality build`` prepares a reference-library folder (the SMILES, a
metadata file and the connectivity keys of ``ref_match``) from a SMILES
library CSV. It is a release tool — ordinary users should not need to run it.
Use it to produce a new canonical library for the next major release, or to
build a non-canonical library for internal testing and fit against it::

    eosquality build --input library.csv --output /tmp/lib/ --name my_test_library
    EOSQUALITY_REFERENCE_LIBRARY_PATH=/tmp/lib/ eosquality fit -r reference_eos4e40_v1.csv -a artifacts_eos4e40_v1/
"""

import importlib.metadata

import click

from eosquality.cli.build import build
from eosquality.cli.fit import fit
from eosquality.cli.run import run
from eosquality.cli.setup import setup

try:
    _VERSION = importlib.metadata.version("eosquality")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"


# Commands for maintainers, listed in their own help block.
DEVELOPER_COMMANDS = ("build",)


class _SectionedGroup(click.Group):
    """Click group whose help lists user and developer commands separately."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Commands in the order they were added.

        Parameters
        ----------
        ctx : click.Context
            Current context.

        Returns
        -------
        list of str
            Command names.
        """
        return list(self.commands)

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        """Write the "Commands" and "Developer commands" help blocks.

        Parameters
        ----------
        ctx : click.Context
            Current context.
        formatter : click.HelpFormatter
            Help formatter to write into.
        """
        rows = {
            name: (name, cmd.get_short_help_str(limit=formatter.width))
            for name, cmd in self.commands.items()
            if not cmd.hidden
        }
        user = [r for n, r in rows.items() if n not in DEVELOPER_COMMANDS]
        dev = [r for n, r in rows.items() if n in DEVELOPER_COMMANDS]
        if user:
            with formatter.section("Commands"):
                formatter.write_dl(user)
        if dev:
            with formatter.section("Developer commands (maintainers only)"):
                formatter.write_dl(dev)


@click.group(
    cls=_SectionedGroup,
    help="Assess the quality of Ersilia model predictions.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(_VERSION, prog_name="eosquality")
def cli() -> None:
    """Every subcommand prints curated progress (``-v`` adds DEBUG logs)."""


# In workflow order: setup once, then fit per model, then run per query set.
cli.add_command(setup)
cli.add_command(fit)
cli.add_command(run)
cli.add_command(build)


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
