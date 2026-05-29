"""Top-level command-line interface for eosquality.

Each subcommand lives in its own module under this package:

- :mod:`eosquality.cli.build` registers ``build``.
- :mod:`eosquality.cli.download` registers ``download``.
- :mod:`eosquality.cli.fit` registers ``fit``.
- :mod:`eosquality.cli.run` registers ``run``.

This module is just the dispatcher: it builds the argparse tree by
calling each submodule's ``register_subparsers``, parses ``sys.argv``,
and invokes the selected command.

End-user workflow
-----------------
Each release is pinned to a canonical reference library, auto-resolved
from env override → ``./data/indices/ersilia_reference_library_vN/`` →
``~/.eosquality/`` cache → S3 download. The usual path is just fit then
run::

    eosquality fit --input eos4e40_v1.csv --output artifacts/ [--k 5]
    eosquality run --input query.csv --artifacts artifacts/ --output scores.csv

Prefetch the library explicitly (useful for CI or airgapped setups)::

    eosquality download [--force]

For maintainers / advanced use
------------------------------
``eosquality build`` rebuilds the vector index from a SMILES library CSV.
It is a release tool — ordinary users should not need to run it. Use it
to produce a new canonical library for the next major release, or to
build a non-canonical index for internal testing; set
``EOSQUALITY_REFERENCE_LIBRARY_PATH`` to point ``fit`` at that folder::

    eosquality build --input library.csv --output /tmp/idx/ [--max-k 50]
    EOSQUALITY_REFERENCE_LIBRARY_PATH=/tmp/idx/ \\
        eosquality fit --input eos4e40_v1.csv --output artifacts/
"""

import argparse
import importlib.metadata
import sys

from eosquality.cli.build import register_subparsers as _register_build
from eosquality.cli.download import register_subparsers as _register_download
from eosquality.cli.fit import register_subparsers as _register_fit
from eosquality.cli.run import register_subparsers as _register_run


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``eosquality`` argparse tree."""
    parser = argparse.ArgumentParser(
        prog="eosquality",
        description="Assess the quality of query data against a fitted reference population.",
    )
    try:
        _pkg_version = importlib.metadata.version("eosquality")
    except importlib.metadata.PackageNotFoundError:
        _pkg_version = "unknown"
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_pkg_version}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    _register_build(subparsers)
    _register_download(subparsers)
    _register_fit(subparsers)
    _register_run(subparsers)

    return parser


def main() -> None:
    """Parse ``sys.argv`` and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
