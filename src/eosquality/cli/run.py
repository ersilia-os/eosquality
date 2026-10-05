"""CLI handler for ``eosquality run``."""

import argparse
import json
import os
import pathlib
import sys
import traceback

import pandas as pd

from eosquality import set_verbosity
from eosquality.exceptions import (
    IncompatibleArtifactsError,
    NotFittedError,
    SchemaError,
)
from eosquality.quality import ErsiliaQuality


def _print_error(message: str, exc: Exception, *, verbose: bool) -> None:
    """Emit a single-line user-facing error; print full traceback in -v mode."""
    print(f"error: {message}: {exc}", file=sys.stderr)
    if verbose:
        traceback.print_exc(file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    """Argparse handler for ``eosquality run``."""
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
    details_path = args.training_details or _default_details_path(args.output)

    print(f"→ reading query CSV: {args.input}", file=sys.stderr)
    try:
        query = pd.read_csv(args.input)
    except FileNotFoundError as exc:
        _print_error(
            f"query CSV not found at '{args.input}'", exc, verbose=args.verbose
        )
        return 1
    except Exception as exc:
        _print_error(
            f"could not read query CSV '{args.input}'", exc, verbose=args.verbose
        )
        return 1

    try:
        eq = ErsiliaQuality.load(args.artifacts)
    except FileNotFoundError as exc:
        _print_error(
            f"artifact at '{args.artifacts}' is incomplete — refit may be required",
            exc,
            verbose=args.verbose,
        )
        return 1
    except IncompatibleArtifactsError as exc:
        _print_error(
            f"artifact at '{args.artifacts}' is not compatible with this eosquality install",
            exc,
            verbose=args.verbose,
        )
        return 1
    except json.JSONDecodeError as exc:
        _print_error(
            f"artifact at '{args.artifacts}' has a malformed JSON file",
            exc,
            verbose=args.verbose,
        )
        return 1
    except Exception as exc:
        _print_error(
            f"could not load artifacts from '{args.artifacts}'",
            exc,
            verbose=args.verbose,
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
        if result.training_details is not None and os.path.exists(details_path):
            print(
                f"error: training details path '{details_path}' already exists; "
                "delete or move it, or pass --training-details.",
                file=sys.stderr,
            )
            return 1
        print(f"→ writing scores: {args.output}", file=sys.stderr)
        output_df.to_csv(args.output, index=False)
        if result.training_details is not None:
            print(f"→ writing training details: {details_path}", file=sys.stderr)
            result.training_details.to_csv(details_path, index=False)
    except (SchemaError, NotFittedError) as exc:
        _print_error(
            "query does not match the fitted reference", exc, verbose=args.verbose
        )
        return 1
    except Exception as exc:
        _print_error("run failed", exc, verbose=args.verbose)
        return 1

    if not args.verbose:
        print(f"Scores saved → {args.output}")
    return 0


def _default_details_path(output: str) -> str:
    """``scores.csv`` → ``scores.training_details.csv``."""
    p = pathlib.Path(output)
    return str(p.with_name(f"{p.stem}.training_details{p.suffix or '.csv'}"))


def register_subparsers(subparsers) -> None:
    """Attach the ``run`` subcommand to *subparsers*."""
    run_p = subparsers.add_parser(
        "run",
        help="Score query data against a fitted reference.",
        description=(
            "Score query molecules with every score in the artifacts. The output "
            "CSV has the query's 'key' and 'input' columns, then each fitted "
            "score with its '*_raw' companion (reference modality: typicality, "
            "extremity, support + support_log, consistency, signal; training "
            "modality: training_domain, training_n_columns, in_training_any). "
            "If the artifacts hold a training modality, a second CSV with one "
            "row per (query, output column) and the nearest training molecules "
            "is written next to it."
        ),
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
        "--training-details",
        default=None,
        dest="training_details",
        metavar="PATH",
        help=(
            "Where to write the per-column training details (default: "
            "<output stem>.training_details.csv). Only written when the "
            "artifacts hold a training modality."
        ),
    )
    run_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print debug messages and diagnostic tables.",
    )
    run_p.set_defaults(func=cmd_run)
