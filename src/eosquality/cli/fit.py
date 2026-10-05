"""CLI handler for ``eosquality fit``."""

import argparse
import os
import pathlib
import sys
import traceback

import pandas as pd

from eosquality import set_verbosity
from eosquality.exceptions import SchemaError
from eosquality.quality import (
    ALL_SCORES,
    DEFAULT_SCORES,
    MIN_REFERENCE_SAMPLES,
    ErsiliaQuality,
)
from eosquality.shared.fit import DEFAULT_MAX_FEATURES
from eosquality.utils.identifiers import extract_from_path, find_eos_id


def _parse_scores(s: str) -> list[str]:
    """Parse a comma-separated --scores arg into a list of score names."""
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _print_error(message: str, exc: Exception, *, verbose: bool) -> None:
    """Emit a single-line user-facing error; print full traceback in -v mode."""
    print(f"error: {message}: {exc}", file=sys.stderr)
    if verbose:
        traceback.print_exc(file=sys.stderr)


def _model_id_from_name(path: str, fallback_version: str) -> tuple[str, str] | None:
    """(eos_id, version) from a file or folder name, or None if absent."""
    name = pathlib.Path(path).name
    try:
        return extract_from_path(name)
    except ValueError:
        eos_id = find_eos_id(name)
        return (eos_id, fallback_version) if eos_id else None


def cmd_fit(args: argparse.Namespace) -> int:
    """Argparse handler for ``eosquality fit``."""
    if args.verbose:
        set_verbosity(True)

    if args.reference is None and args.training is None:
        print("error: give --reference, --training, or both.", file=sys.stderr)
        return 1
    if args.artifacts is not None:
        if args.reference is not None or args.output is not None:
            print(
                "error: --artifacts adds training sets to an existing artifacts "
                "folder; it cannot be combined with --reference or --output.",
                file=sys.stderr,
            )
            return 1
        if args.training is None:
            print("error: --artifacts needs --training.", file=sys.stderr)
            return 1
        return _add_training(args)
    if args.output is None:
        print(
            "error: --output is required (or --artifacts to add training).",
            file=sys.stderr,
        )
        return 1
    if os.path.exists(args.output):
        print(
            f"error: output path '{args.output}' already exists; "
            "delete or move it before re-running fit.",
            file=sys.stderr,
        )
        return 1

    # Model id: from the reference CSV name, else from the training folder name.
    source = args.reference if args.reference is not None else args.training
    model_id = _model_id_from_name(source, args.version)
    if model_id is None:
        print(
            f"error: could not find a valid EOS identifier in "
            f"'{pathlib.Path(source).name}'. Rename it to include the model ID "
            "and version (e.g. 'eos4e40_v1.csv' or 'training_eos4e40_v1/').",
            file=sys.stderr,
        )
        return 1
    eos_id, version = model_id
    if args.reference is not None and args.training is not None:
        training_id = _model_id_from_name(args.training, version)
        if training_id is not None and training_id[0] != eos_id:
            print(
                f"error: --training is for {training_id[0]} but --reference is "
                f"for {eos_id}.",
                file=sys.stderr,
            )
            return 1

    reference = None
    if args.reference is not None:
        print(f"→ reading reference CSV: {args.reference}", file=sys.stderr)
        try:
            reference = pd.read_csv(args.reference)
        except Exception as exc:
            _print_error(
                f"could not read reference CSV '{args.reference}'",
                exc,
                verbose=args.verbose,
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
            vector_index=args.vector_index,
            ignore_size=args.ignore_size,
            scores=scores,
            max_features=max_features,
            max_signal_train_samples=max_signal_train_samples,
            signal_descriptor=args.signal_descriptor,
            training=args.training,
            training_predictions=args.training_predictions,
        )
        eq.save(args.output)
    except SchemaError as exc:
        _print_error(
            "input does not match the expected schema", exc, verbose=args.verbose
        )
        return 1
    except FileNotFoundError as exc:
        _print_error(
            "fit failed because a required file is missing (the canonical "
            "library? run 'eosquality download')",
            exc,
            verbose=args.verbose,
        )
        return 1
    except Exception as exc:
        _print_error("fit failed", exc, verbose=args.verbose)
        return 1
    return 0


def _add_training(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.artifacts):
        print(
            f"error: artifacts folder '{args.artifacts}' does not exist.",
            file=sys.stderr,
        )
        return 1
    model_id = _model_id_from_name(args.training, args.version)
    try:
        ErsiliaQuality.add_training(
            args.artifacts,
            args.training,
            args.training_predictions,
            eos_id=model_id[0] if model_id else None,
            version=model_id[1] if model_id else None,
        )
    except Exception as exc:
        _print_error("adding training sets failed", exc, verbose=args.verbose)
        return 1
    return 0


def register_subparsers(subparsers) -> None:
    """Attach the ``fit`` subcommand to *subparsers*."""
    fit_p = subparsers.add_parser(
        "fit",
        help="Fit a reference population and save artifacts.",
        description=(
            "Fit quality scores for one model and save the artifacts. Two "
            "modalities, each fitted when its data is given: --reference (the "
            "model's predictions on the reference library) and --training (a "
            "folder of per-output-column training sets). Give either or both; "
            "--training with --artifacts adds training sets to an existing "
            "artifacts folder. The model id is read from the --reference file "
            "name, else from the --training folder name (e.g. eos4e40_v1.csv, "
            "training_eos4e40_v1/). The canonical reference library is resolved "
            "locally (EOSQUALITY_REFERENCE_LIBRARY_PATH → ./data/indices/<library>/ "
            "→ ~/.eosquality/indices/<library>/); fit never downloads."
        ),
    )
    fit_p.add_argument(
        "--reference",
        default=None,
        metavar="CSV",
        help=(
            "Reference modality: the model's predictions on the reference "
            "library ('key', 'input' and one numeric column per output)."
        ),
    )
    fit_p.add_argument(
        "--training",
        default=None,
        metavar="DIR",
        help=(
            "Training modality: folder with one <output_column>.csv per column "
            "('smiles', optional 'y', optional 'key')."
        ),
    )
    fit_p.add_argument(
        "--training-predictions",
        default=None,
        dest="training_predictions",
        metavar="CSV",
        help=(
            "The model's own predictions on the training molecules (Ersilia "
            "output CSV), stored with the training modality."
        ),
    )
    fit_p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="New folder for the saved artifacts (must not exist).",
    )
    fit_p.add_argument(
        "--artifacts",
        "-a",
        default=None,
        metavar="PATH",
        help=(
            "Existing artifacts folder to add the --training modality to, in "
            "place (reference files are left untouched)."
        ),
    )
    fit_p.add_argument(
        "--vector-index",
        default=None,
        dest="vector_index",
        metavar="PATH",
        help=(
            "Fit against a non-canonical vector index folder built with "
            "'eosquality build' (default: the canonical reference library). "
            "The folder's absolute path is recorded in the artifacts and must "
            "still exist at run time."
        ),
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
            f"Skip the minimum-row check ({MIN_REFERENCE_SAMPLES:,} rows required). "
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
            "Cap on the number of training rows the 'signal' XGBoost model is "
            "fit on (default: 1000, for fast iteration). Pass 0 or a negative "
            "value to use the full training slice. Calibration always uses the "
            "full validation slice. Ignored when 'signal' is not in the score set."
        ),
    )
    fit_p.add_argument(
        "--signal-descriptor",
        default="physchem",
        choices=("physchem", "maccs"),
        dest="signal_descriptor",
        help=(
            "Feature backend the 'signal' score uses (default: physchem). "
            "'physchem' = RDKit physicochemical descriptors; 'maccs' = MACCS "
            "structural keys. Both are precomputed in the library. The choice "
            "is recorded in the saved artifact and used by 'eosquality run'. "
            "Ignored when 'signal' is not in the score set."
        ),
    )
    fit_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print debug messages and diagnostic tables.",
    )
    fit_p.set_defaults(func=cmd_fit)
