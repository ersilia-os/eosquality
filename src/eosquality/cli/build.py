"""CLI handler for ``eosquality build`` — build a vector index from a SMILES CSV.

Release / maintenance tool. End users do not normally call this; the
canonical reference library ships with each release. Used to prepare a
replacement library for the next release, or to build a non-canonical
index for internal testing (pass the result to ``fit --vector-index``).
"""

import argparse
import pathlib
import sys

import pandas as pd

from eosquality import set_verbosity
from eosquality.vectorindex import VectorIndex


def cmd_build(args: argparse.Namespace) -> int:
    """Argparse handler for ``eosquality build``."""
    if args.verbose:
        set_verbosity(True)

    try:
        df = pd.read_csv(args.input)
    except Exception as exc:
        print(
            f"error: could not read library file '{args.input}': {exc}", file=sys.stderr
        )
        return 1

    if "smiles" not in df.columns:
        print(
            f"error: library CSV must contain a 'smiles' column "
            f"(found: {list(df.columns)})",
            file=sys.stderr,
        )
        return 1

    smiles = list(df["smiles"])
    print(f"Building vector index for {len(smiles):,} molecules → {args.output}")

    try:
        VectorIndex.build(
            smiles=smiles,
            output_dir=args.output,
            max_k=args.max_k,
            radius=args.radius,
            n_bits=args.n_bits,
            verbose=args.verbose,
            library_name=pathlib.Path(args.input).stem,
            max_samples=args.max_samples,
        )
    except Exception as exc:
        print(f"error: index build failed: {exc}", file=sys.stderr)
        return 1

    if not args.verbose:
        print(f"Vector index saved → {args.output}")
    return 0


def register_subparsers(subparsers) -> None:
    """Attach the ``index`` subcommand to *subparsers*."""
    build_p = subparsers.add_parser(
        "build",
        help="(release tool) Build a vector index from a reference library CSV.",
        description=(
            "Build a Morgan-fingerprint kNN index for a SMILES library. "
            "This is a release / maintenance tool used to produce the canonical "
            "reference library that ships with each major version of eosquality — "
            "end users do not normally need to run it. Use it to prepare a "
            "replacement library for the next release, or to build a non-canonical "
            "index for internal testing (pass the result to 'fit --vector-index')."
        ),
    )
    build_p.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Path to the reference library CSV file (must have a 'smiles' column).",
    )
    build_p.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="PATH",
        help="Output folder for the vector index (e.g. data/indices/ersilia_reference_library/).",
    )
    build_p.add_argument(
        "--max-k",
        type=int,
        default=50,
        dest="max_k",
        metavar="K",
        help="Maximum k to pre-compute for self-kNN (default: 50).",
    )
    build_p.add_argument(
        "--radius",
        type=int,
        default=2,
        metavar="R",
        help="Morgan radius (default: 2).",
    )
    build_p.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        dest="n_bits",
        metavar="N",
        help="Number of bits in the Morgan vector (default: 2048).",
    )
    build_p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        metavar="N",
        help="Truncate input to the first N molecules (for testing).",
    )
    build_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information.",
    )
    build_p.set_defaults(func=cmd_build)
