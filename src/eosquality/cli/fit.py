"""CLI handler for the fit submodule: ``fit``."""

import argparse
import os
import sys

import pandas as pd

from eosquality import set_verbosity
from eosquality.quality import ErsiliaQuality
from eosquality.utils.identifiers import extract_from_path


def cmd_fit(args: argparse.Namespace) -> int:
    if args.verbose:
        set_verbosity(True)

    if os.path.exists(args.output):
        print(
            f"error: output path '{args.output}' already exists; "
            "delete or move it before re-running fit.",
            file=sys.stderr,
        )
        return 1

    # Extract eos_id and version from the source filename.
    # Version falls back to --version flag if not found in the filename.
    try:
        eos_id, version = extract_from_path(args.input)
    except ValueError:
        try:
            from eosquality.utils.identifiers import _EOS_ID_ANYWHERE_RE

            basename = os.path.basename(args.input)
            m = _EOS_ID_ANYWHERE_RE.search(basename)
            if m:
                eos_id = m.group(1)
                version = args.version
            else:
                print(
                    f"error: could not find a valid EOS identifier in filename "
                    f"'{os.path.basename(args.input)}'. "
                    "Rename the file to include the model ID and version "
                    "(e.g. 'eos4e40_v1.csv').",
                    file=sys.stderr,
                )
                return 1
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        reference = pd.read_csv(args.input)
    except Exception as exc:
        print(
            f"error: could not read reference file '{args.input}': {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        eq = ErsiliaQuality(k=args.k, verbose=args.verbose)
        eq.fit(
            reference,
            eos_id=eos_id,
            version=version,
            ignore_size=args.ignore_size,
        )
        eq.save(args.output)
    except Exception as exc:
        print(f"error: fit failed: {exc}", file=sys.stderr)
        return 1

    if not args.verbose:
        print(f"Artifacts saved → {args.output}")
    return 0


def register_subparsers(subparsers) -> None:
    """Attach the ``fit`` subcommand to *subparsers*."""
    fit_p = subparsers.add_parser(
        "fit",
        help="Fit a reference population and save artifacts.",
        description=(
            "Fit a reference population from a CSV file and persist the artifacts. "
            "The reference library is resolved automatically (env override → "
            "./data/indices/ersilia_reference_library_vN/ → ~/.eosquality/ "
            "cache → S3 download); set EOSQUALITY_REFERENCE_LIBRARY_PATH to "
            "point at a non-canonical folder for internal testing."
        ),
    )
    fit_p.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Path to the reference CSV file (must contain 'key', 'input', and numeric feature columns).",
    )
    fit_p.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="PATH",
        help="Output folder for the saved artifacts (e.g. artifacts/).",
    )
    fit_p.add_argument(
        "--k",
        type=int,
        default=5,
        metavar="K",
        help="Number of nearest neighbors (default: 5).",
    )
    fit_p.add_argument(
        "--version",
        default="v1",
        metavar="VERSION",
        help=(
            "Dataset version (e.g. 'v1'). Used only when the version cannot be "
            "extracted from the filename. If the filename contains a version "
            "(e.g. 'eos4e40_v2.csv'), the filename always wins (default: v1)."
        ),
    )
    fit_p.add_argument(
        "--ignore-size",
        action="store_true",
        dest="ignore_size",
        help=(
            f"Skip the minimum-row check ({10_000:,} rows required). "
            "For development and testing only."
        ),
    )
    fit_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print informative progress and diagnostic tables.",
    )
    fit_p.set_defaults(func=cmd_fit)
