"""``eosquality fit`` — fit the reference and/or training modality."""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING

import click

from eosquality._registry import (
    ALL_SCORES,
    DEFAULT_MAX_FEATURES,
    DEFAULT_SCORES,
    MIN_REFERENCE_SAMPLES,
)
from eosquality.cli._common import (
    CliError,
    require_new_path,
    run_command,
    staged_log,
    verbose_option,
)
from eosquality.utils import console
from eosquality.utils.identifiers import extract_from_path, find_eos_id
from eosquality.utils.logging import logger

if TYPE_CHECKING:  # heavy imports happen inside the command, not at CLI start-up
    import pandas as pd

LOG_FILE = "eosquality.log"


def model_id_from_name(path: str, fallback_version: str) -> tuple[str, str] | None:
    """Read ``(eos_id, version)`` from a file or folder name.

    Parameters
    ----------
    path : str
        File or folder path, e.g. ``eos4e40_v1.csv`` or ``training_eos4e40_v1/``.
    fallback_version : str
        Version to use when the name carries an EOS id but no version.

    Returns
    -------
    tuple of (str, str) or None
        The model id and version, or ``None`` if the name has no EOS id.
    """
    name = pathlib.Path(path).name
    try:
        return extract_from_path(name)
    except ValueError:
        eos_id = find_eos_id(name)
        return (eos_id, fallback_version) if eos_id else None


def _resolve_model_id(reference: str | None, training_sets: str | None, version: str):
    source = reference if reference is not None else training_sets
    model_id = model_id_from_name(source, version)
    if model_id is None:
        raise CliError(
            f"could not find a valid EOS identifier in '{pathlib.Path(source).name}'. "
            "Rename it to include the model ID and version (e.g. 'eos4e40_v1.csv' "
            "or 'training_eos4e40_v1/')."
        )
    if reference is not None and training_sets is not None:
        training_id = model_id_from_name(training_sets, model_id[1])
        if training_id is not None and training_id[0] != model_id[0]:
            raise CliError(
                f"--training-sets is for {training_id[0]} but --reference is for "
                f"{model_id[0]}."
            )
    return model_id


def _check_inputs(reference, training_sets, output, artifacts) -> None:
    if reference is None and training_sets is None:
        raise CliError("give --reference, --training-sets, or both.")
    if artifacts is not None:
        if reference is not None or output is not None:
            raise CliError(
                "--artifacts adds training sets to an existing artifacts folder; "
                "it cannot be combined with --reference or --output."
            )
        if training_sets is None:
            raise CliError("--artifacts needs --training-sets.")
        if not pathlib.Path(artifacts).is_dir():
            raise CliError(f"artifacts folder '{artifacts}' does not exist.")
    elif output is None:
        raise CliError("--output is required (or --artifacts to add training sets).")


def _read_reference(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    logger.info(f"fit | reading reference CSV {path}")
    import pandas as pd

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise CliError(f"could not read reference CSV '{path}': {exc}") from exc


def _positive_or_none(value: int) -> int | None:
    return value if value > 0 else None


@click.command(
    "fit",
    help=(
        "Fit quality scores for one model and save the artifacts. Two modalities, "
        "each fitted when its data is given: --reference (the model's predictions "
        "on the reference library) and --training-sets (a folder of per-output-column "
        "training sets). Give either or both; --training-sets with --artifacts adds "
        "training sets to an existing artifacts folder. The model id is read from "
        "the --reference file name, else from the --training-sets folder name (e.g. "
        "eos4e40_v1.csv, training_eos4e40_v1/). The canonical reference library "
        "is resolved locally; fit never downloads."
    ),
    short_help="Fit quality scores and save artifacts.",
)
@click.option(
    "--reference",
    metavar="CSV",
    help="Reference modality: predictions on the reference library.",
)
@click.option(
    "--training-sets",
    metavar="DIR",
    help=(
        "Training modality: folder with one <output_column>.csv per column "
        "('smiles', optional 'y', optional 'key')."
    ),
)
@click.option(
    "--training-predictions",
    metavar="CSV",
    help="The model's own predictions on the training molecules (Ersilia output CSV).",
)
@click.option(
    "--output", "-o", metavar="PATH", help="New artifacts folder (must not exist)."
)
@click.option(
    "--artifacts",
    "-a",
    metavar="PATH",
    help="Existing artifacts folder to add --training-sets to, in place.",
)
@click.option(
    "--vector-index",
    metavar="PATH",
    help="Fit the reference modality against a non-canonical vector index folder.",
)
@click.option(
    "--k",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    metavar="K",
    help="Nearest neighbours.",
)
@click.option(
    "--version",
    default="v1",
    show_default=True,
    metavar="VERSION",
    help="Dataset version, used only if the file/folder name has none.",
)
@click.option(
    "--ignore-size",
    is_flag=True,
    help=f"Skip the {MIN_REFERENCE_SAMPLES:,}-row minimum (testing only).",
)
@click.option(
    "--max-features",
    default=DEFAULT_MAX_FEATURES,
    show_default=True,
    metavar="N",
    help="Feature-selection cap; 0 or negative disables it.",
)
@click.option(
    "--scores",
    metavar="LIST",
    help=(
        f"Comma-separated reference scores to fit, from: {', '.join(ALL_SCORES)}. "
        f"Default: {','.join(DEFAULT_SCORES)} ('signal' is opt-in)."
    ),
)
@click.option(
    "--max-signal-samples",
    default=1000,
    show_default=True,
    metavar="N",
    help="Signal training rows; 0 or negative uses the full train slice.",
)
@click.option(
    "--signal-descriptor",
    default="physchem",
    show_default=True,
    type=click.Choice(["physchem", "maccs"]),
    help="Feature backend of the 'signal' score.",
)
@verbose_option
def fit(
    reference: str | None,
    training_sets: str | None,
    training_predictions: str | None,
    output: str | None,
    artifacts: str | None,
    vector_index: str | None,
    k: int,
    version: str,
    ignore_size: bool,
    max_features: int,
    scores: str | None,
    max_signal_samples: int,
    signal_descriptor: str,
    verbose: bool,
) -> None:
    """Fit the reference and/or training modality and save the artifacts.

    Parameters
    ----------
    reference : str or None
        Reference predictions CSV.
    training_sets : str or None
        Training-set folder.
    training_predictions : str or None
        Model predictions on the training molecules.
    output : str or None
        New artifacts folder.
    artifacts : str or None
        Existing artifacts folder to add the training sets to.
    vector_index : str or None
        Custom vector index for the reference modality.
    k : int
        Nearest neighbours.
    version : str
        Fallback dataset version.
    ignore_size : bool
        Skip the minimum reference size.
    max_features : int
        Feature-selection cap (non-positive disables it).
    scores : str or None
        Comma-separated reference scores.
    max_signal_samples : int
        Signal training rows (non-positive: all).
    signal_descriptor : str
        Signal descriptor backend.
    verbose : bool
        Print debug messages and diagnostic tables.
    """

    options = dict(locals())
    run_command(lambda: _fit(**options), verbose=verbose, command="fit")


def _fit(
    *,
    reference,
    training_sets,
    training_predictions,
    output,
    artifacts,
    vector_index,
    k,
    version,
    ignore_size,
    max_features,
    scores,
    max_signal_samples,
    signal_descriptor,
    verbose,
) -> None:
    _check_inputs(reference, training_sets, output, artifacts)
    started = time.perf_counter()
    if artifacts is not None:
        _add_training(artifacts, training_sets, training_predictions, version, started)
        return
    require_new_path(output)
    eos_id, model_version = _resolve_model_id(reference, training_sets, version)
    score_list = (
        [t.strip() for t in scores.split(",") if t.strip()]
        if scores
        else list(DEFAULT_SCORES)
    )
    output_path = pathlib.Path(output)
    console.summary_panel(
        "eosquality · fit",
        [
            ("model", f"{eos_id} {model_version}"),
            ("reference", console.path(reference) if reference else "—"),
            ("training sets", console.path(training_sets) if training_sets else "—"),
            ("scores", ", ".join(score_list) if reference else "—"),
            ("output", console.path(output_path)),
        ],
        icon="◆",
    )
    with staged_log(output_path / LOG_FILE) as log_path:
        logger.info(f"fit | eosquality {eos_id} {model_version} → {output_path}")
        from eosquality.quality import ErsiliaQuality

        eq = ErsiliaQuality(k=k, verbose=verbose)
        eq.fit(
            _read_reference(reference),
            eos_id=eos_id,
            version=model_version,
            vector_index=vector_index,
            ignore_size=ignore_size,
            scores=score_list,
            max_features=_positive_or_none(max_features),
            max_signal_train_samples=_positive_or_none(max_signal_samples),
            signal_descriptor=signal_descriptor,
            training_sets=training_sets,
            training_predictions=training_predictions,
        )
        with console.section("Save") as section:
            eq.save(output_path)
            section.summary = f"{console.folder_size(output_path)}"
    _fit_summary(eq, output_path, log_path, started)


def _add_training(artifacts, training_sets, training_predictions, version, started):
    """``fit --artifacts``: add the training modality to existing artifacts."""
    model_id = model_id_from_name(training_sets, version)
    folder = pathlib.Path(artifacts)
    console.summary_panel(
        "eosquality · fit (add training)",
        [
            ("artifacts", console.path(folder)),
            ("training sets", console.path(training_sets)),
        ],
        icon="◆",
    )
    with logger.log_file(folder / LOG_FILE) as log_path:
        logger.info(f"fit | adding training sets {training_sets} → {folder}")
        from eosquality.quality import ErsiliaQuality

        eq = ErsiliaQuality.add_training(
            artifacts,
            training_sets,
            training_predictions,
            eos_id=model_id[0] if model_id else None,
            version=model_id[1] if model_id else None,
        )
    _fit_summary(eq, folder, log_path, started)


def _fit_summary(eq, folder: pathlib.Path, log_path: pathlib.Path, started) -> None:
    """Final ``✓ Fit complete`` panel."""
    eos_id, version = eq._model_id()
    scores = list(eq._components()) + list(eq._training_components())
    console.summary_panel(
        "Fit complete",
        [
            ("model", f"{eos_id} {version}"),
            ("modalities", " + ".join(eq.modalities_)),
            ("scores", ", ".join(scores)),
            (
                "artifacts",
                f"{console.path(folder)}  [dim]{console.folder_size(folder)}[/]",
            ),
            ("log", console.path(log_path)),
            ("time", console.elapsed(time.perf_counter() - started)),
        ],
        color="green",
        icon="✓",
    )
