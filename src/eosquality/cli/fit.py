"""CLI handler for ``eosquality fit``."""

import argparse
import os
import sys
import traceback

import pandas as pd

from eosquality import set_verbosity
from eosquality.exceptions import SchemaError
from eosquality.quality import ALL_SCORES, DEFAULT_SCORES, ErsiliaQuality
from eosquality.shared.fit import DEFAULT_MAX_FEATURES
from eosquality.utils.identifiers import extract_from_path


def _parse_scores(s: str) -> list[str]:
    """Parse a comma-separated --scores arg into a list of score names."""
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _print_error(message: str, exc: Exception, *, verbose: bool) -> None:
    """Emit a single-line user-facing error; print full traceback in -v mode."""
    print(f"error: {message}: {exc}", file=sys.stderr)
    if verbose:
        traceback.print_exc(file=sys.stderr)


def cmd_fit(args: argparse.Namespace) -> int:
    """Argparse handler for ``eosquality fit``."""
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

    print(f"→ reading reference CSV: {args.input}", file=sys.stderr)
    try:
        reference = pd.read_csv(args.input)
    except FileNotFoundError as exc:
        _print_error(
            f"reference CSV not found at '{args.input}'", exc, verbose=args.verbose
        )
        return 1
    except Exception as exc:
        _print_error(
            f"could not read reference CSV '{args.input}'", exc, verbose=args.verbose
        )
        return 1

    max_features = args.max_features if args.max_features > 0 else None
    max_signal_train_samples = (
        args.max_signal_samples if args.max_signal_samples > 0 else None
    )

    scores = _parse_scores(args.scores) if args.scores else list(DEFAULT_SCORES)

    try:
        eq = ErsiliaQuality(k=args.k, verbose=args.verbose)
        eq.fit(
            reference,
            eos_id=eos_id,
            version=version,
            ignore_size=args.ignore_size,
            scores=scores,
            max_features=max_features,
            max_signal_train_samples=max_signal_train_samples,
            signal_descriptor=args.signal_descriptor,
        )
        eq.save(args.output)
    except SchemaError as exc:
        _print_error(
            "reference does not match the expected schema", exc, verbose=args.verbose
        )
        return 1
    except FileNotFoundError as exc:
        _print_error(
            "fit failed because a required file is missing (likely the canonical "
            "library — run 'eosquality download' first)",
            exc,
            verbose=args.verbose,
        )
        return 1
    except ValueError as exc:
        _print_error("fit failed", exc, verbose=args.verbose)
        return 1
    except Exception as exc:
        _print_error("fit failed", exc, verbose=args.verbose)
        return 1

    return 0


def register_subparsers(subparsers) -> None:
    """Attach the ``fit`` subcommand to *subparsers*."""
    fit_p = subparsers.add_parser(
        "fit",
        help="Fit a reference population and save artifacts.",
        description=(
            "Fit a reference population from a CSV file and persist the artifacts. "
            "The reference library is resolved locally (env override → "
            "./data/indices/ersilia_reference_library_vN/ → ~/.eosquality/ "
            "cache); fit never downloads — run 'eosquality download' first if "
            "the library isn't cached yet. Set EOSQUALITY_REFERENCE_LIBRARY_PATH "
            "to point at a non-canonical folder for internal testing."
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
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        dest="max_features",
        metavar="N",
        help=(
            "Cap on the number of features kept after correlation-cluster "
            f"medoid reduction (default: {DEFAULT_MAX_FEATURES}). "
            "Pass 0 or a negative value to disable reduction."
        ),
    )
    fit_p.add_argument(
        "--scores",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of scores to fit. Choices: "
            f"{', '.join(ALL_SCORES)}. "
            "Example: --scores signal,typicality. "
            f"Default: {','.join(DEFAULT_SCORES)} ('signal' is opt-in)."
        ),
    )
    fit_p.add_argument(
        "--max-signal-samples",
        type=int,
        default=1000,
        dest="max_signal_samples",
        metavar="N",
        help=(
            "Cap on the number of training rows the 'signal' score actually "
            "fits its XGBoost models on (default: 1000, for fast iteration). "
            "Pass 0 or a negative value to use the full training slice. "
            "The validation slice is never subsampled. Ignored when 'signal' "
            "is not in the score set."
        ),
    )
    fit_p.add_argument(
        "--signal-descriptor",
        default="physchem",
        choices=("physchem", "maccs"),
        dest="signal_descriptor",
        help=(
            "Feature backend the 'signal' score uses (default: physchem). "
            "'physchem' = 217 RDKit physicochemical descriptors (precomputed "
            "in the library). 'maccs' = 167-bit RDKit MACCS fingerprint "
            "(computed on demand). The chosen descriptor is recorded in the "
            "saved artifact; 'eosquality run' uses whichever was set at fit "
            "time. Ignored when 'signal' is not in the score set."
        ),
    )
    fit_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print informative progress and diagnostic tables.",
    )
    fit_p.set_defaults(func=cmd_fit)
