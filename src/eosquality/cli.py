"""Command-line interface for eosquality.

End-user workflow
-----------------
Each release is pinned to a canonical reference library, auto-resolved from
env override → ./data/indices/ersilia_reference_library_vN/ → ~/.eosquality/
cache → S3 download. The usual path is just fit then run:

    eosquality fit --input eos4e40_v1.csv --output artifacts/ [--k 20]
    eosquality run --input query.csv --artifacts artifacts/ --output scores.csv

Prefetch the library explicitly (useful for CI or airgapped setups):

    eosquality download [--force]

For maintainers / advanced use
------------------------------
``eosquality index`` rebuilds the vector index from a SMILES library CSV.
It is a release tool — ordinary users should not need to run it. Use it to
produce a new canonical library for the next major release, or to build a
non-canonical index for internal testing; set
``EOSQUALITY_REFERENCE_LIBRARY_PATH`` to point ``fit`` at that folder.

    eosquality index --input library.csv --output /tmp/idx/ [--max-k 50]
    EOSQUALITY_REFERENCE_LIBRARY_PATH=/tmp/idx/ \\
        eosquality fit --input eos4e40_v1.csv --output artifacts/
"""

import argparse
import importlib.metadata
import pathlib
import sys

import pandas as pd

from eosquality import set_verbosity
from eosquality._library import reference_library_csv_path, reference_library_path
from eosquality.quality.api import ErsiliaQuality
from eosquality.utils.identifiers import extract_from_path
from eosquality.vectorindex.backend import VectorIndex


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    if args.verbose:
        set_verbosity(True)

    try:
        df = pd.read_csv(args.input)
    except Exception as exc:
        print(f"error: could not read library file '{args.input}': {exc}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def cmd_fit(args: argparse.Namespace) -> int:
    if args.verbose:
        set_verbosity(True)

    # Extract eos_id and version from the source filename.
    # Version falls back to --version flag if not found in the filename.
    try:
        eos_id, version = extract_from_path(args.input)
    except ValueError:
        # Version may be absent from filename — try eos_id alone
        try:
            from eosquality.utils.identifiers import _EOS_ID_ANYWHERE_RE
            import os
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
        print(f"error: could not read reference file '{args.input}': {exc}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace) -> int:
    if args.verbose:
        set_verbosity(True)
    try:
        csv_path = reference_library_csv_path(force_download=args.force)
        index_path = reference_library_path(force_download=args.force)
    except Exception as exc:
        print(f"error: could not download reference library: {exc}", file=sys.stderr)
        return 1
    print(f"Library CSV    → {csv_path}")
    print(f"Library index  → {index_path}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    if args.verbose:
        set_verbosity(True)

    try:
        query = pd.read_csv(args.input)
    except Exception as exc:
        print(f"error: could not read query file '{args.input}': {exc}", file=sys.stderr)
        return 1

    try:
        eq = ErsiliaQuality.load(args.artifacts)
        if args.verbose:
            eq.verbose = True
    except Exception as exc:
        print(f"error: could not load artifacts from '{args.artifacts}': {exc}", file=sys.stderr)
        return 1

    try:
        result = eq.run(query)
        scores = result.scores.drop(columns=["nearest_reference_ids"], errors="ignore")
        scores.to_csv(args.output, index=True)
    except Exception as exc:
        print(f"error: run failed: {exc}", file=sys.stderr)
        return 1

    if not args.verbose:
        print(f"Scores saved → {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
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

    # ---- index ----
    index_p = subparsers.add_parser(
        "index",
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
    index_p.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to the reference library CSV file (must have a 'smiles' column).",
    )
    index_p.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output folder for the vector index (e.g. data/indices/ersilia_reference_library/).",
    )
    index_p.add_argument(
        "--max-k",
        type=int,
        default=50,
        dest="max_k",
        metavar="K",
        help="Maximum k to pre-compute for self-kNN (default: 50).",
    )
    index_p.add_argument(
        "--radius",
        type=int,
        default=2,
        metavar="R",
        help="Morgan radius (default: 2).",
    )
    index_p.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        dest="n_bits",
        metavar="N",
        help="Number of bits in the Morgan vector (default: 2048).",
    )
    index_p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        metavar="N",
        help="Truncate input to the first N molecules (for testing).",
    )
    index_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information.",
    )
    index_p.set_defaults(func=cmd_index)

    # ---- fit ----
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
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to the reference CSV file (must contain 'key', 'input', and numeric feature columns).",
    )
    fit_p.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output folder for the saved artifacts (e.g. artifacts/).",
    )
    fit_p.add_argument(
        "--k",
        type=int,
        default=20,
        metavar="K",
        help="Number of nearest neighbors (default: 20).",
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
        "--verbose", "-v",
        action="store_true",
        help="Print informative progress and diagnostic tables.",
    )
    fit_p.set_defaults(func=cmd_fit)

    # ---- download ----
    download_p = subparsers.add_parser(
        "download",
        help="Prefetch the canonical reference library into the local cache.",
        description=(
            "Explicitly download the reference library for this release of "
            "eosquality into ~/.eosquality/indices/. Useful for CI or "
            "airgapped setups where you want to fetch before the first fit. "
            "If a valid cached copy already exists, this is a no-op unless "
            "--force is passed. Honors EOSQUALITY_REFERENCE_BASE_URL for "
            "staging buckets and EOSQUALITY_REFERENCE_LIBRARY_PATH to short-"
            "circuit to an existing folder."
        ),
    )
    download_p.add_argument(
        "--force", "-f",
        action="store_true",
        help="Redownload even if a valid cached copy already exists.",
    )
    download_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information.",
    )
    download_p.set_defaults(func=cmd_download)

    # ---- run ----
    run_p = subparsers.add_parser(
        "run",
        help="Score query data against a fitted reference.",
        description="Score query samples against previously fitted reference artifacts.",
    )
    run_p.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to the query CSV file.",
    )
    run_p.add_argument(
        "--artifacts", "-a",
        required=True,
        metavar="PATH",
        help="Path to the fitted artifacts produced by 'eosquality fit'.",
    )
    run_p.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output path for the scores CSV file.",
    )
    run_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print informative progress and diagnostic tables.",
    )
    run_p.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
