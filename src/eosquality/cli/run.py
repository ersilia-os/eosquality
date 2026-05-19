"""CLI handler for the run submodule: ``run``."""

import argparse
import os
import sys

import pandas as pd

from eosquality import set_verbosity
from eosquality.quality import ErsiliaQuality


def cmd_run(args: argparse.Namespace) -> int:
    if args.verbose:
        set_verbosity(True)

    if not os.path.isdir(args.artifacts):
        print(
            f"error: artifacts folder '{args.artifacts}' does not exist.",
            file=sys.stderr,
        )
        return 1

    if os.path.exists(args.output):
        print(
            f"error: output path '{args.output}' already exists; "
            "delete or move it before re-running.",
            file=sys.stderr,
        )
        return 1

    try:
        query = pd.read_csv(args.input)
    except Exception as exc:
        print(
            f"error: could not read query file '{args.input}': {exc}", file=sys.stderr
        )
        return 1

    try:
        eq = ErsiliaQuality.load(args.artifacts)
        if args.verbose:
            eq.verbose = True
    except Exception as exc:
        print(
            f"error: could not load artifacts from '{args.artifacts}': {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        result = eq.run(query)
        prepend = [c for c in ("key", "input") if c in query.columns]
        output_df = pd.concat(
            [
                query[prepend].reset_index(drop=True),
                result.scores.reset_index(drop=True),
            ],
            axis=1,
        )
        output_df.to_csv(args.output, index=False)
    except Exception as exc:
        print(f"error: run failed: {exc}", file=sys.stderr)
        return 1

    if not args.verbose:
        print(f"Scores saved → {args.output}")
    return 0


def register_subparsers(subparsers) -> None:
    """Attach the ``run`` subcommand to *subparsers*."""
    run_p = subparsers.add_parser(
        "run",
        help="Score query data against a fitted reference.",
        description="Score query samples against previously fitted reference artifacts.",
    )
    run_p.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Path to the query CSV file.",
    )
    run_p.add_argument(
        "--artifacts",
        "-a",
        required=True,
        metavar="PATH",
        help="Path to the fitted artifacts produced by 'eosquality fit'.",
    )
    run_p.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="PATH",
        help="Output path for the scores CSV file.",
    )
    run_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print informative progress and diagnostic tables.",
    )
    run_p.set_defaults(func=cmd_run)
