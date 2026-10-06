"""``eosquality fit`` — fit the reference and/or training scores of one model."""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING

import click

from eosquality._registry import (
    ALL_SCORES,
    REFERENCE_SCORES,
    SCORE_NAMES,
    TRAINING_SCORES,
)
from eosquality.cli._common import (
    CliError,
    run_command,
    staged_log,
    verbose_option,
)
from eosquality.utils import console
from eosquality.utils.identifiers import model_from_names
from eosquality.utils.logging import logger

if TYPE_CHECKING:  # heavy imports happen inside the command, not at CLI start-up
    import pandas as pd

LOG_FILE = "eosquality.log"
REFERENCE_DIR = "reference_mode"
TRAINING_DIR = "training_mode"


def parse_exclude(values: tuple[str, ...]) -> list[str]:
    """Score names from repeated and/or comma-separated ``--exclude`` values.

    Parameters
    ----------
    values : tuple of str
        Raw option values.

    Returns
    -------
    list of str
        Score names, in order, without duplicates.

    Raises
    ------
    CliError
        If a name is not a score.
    """
    names = [n.strip() for v in values for n in v.split(",") if n.strip()]
    unknown = [n for n in names if n not in ALL_SCORES]
    if unknown:
        raise CliError(
            f"--exclude: unknown score(s) {', '.join(unknown)}; choose from "
            f"{', '.join(ALL_SCORES)}."
        )
    return list(dict.fromkeys(names))


def resolve_model(reference, training_sets, artifacts) -> tuple[str, str]:
    """The model named by the input and artifacts paths (they must agree).

    Parameters
    ----------
    reference, training_sets : str or None
        ``--reference`` CSV and ``--training-sets`` folder.
    artifacts : str
        ``--artifacts`` folder.

    Returns
    -------
    tuple of (str, str)
        ``(eos_id, version)``.

    Raises
    ------
    CliError
        If a name lacks ``<eos_id>_<version>`` or the names disagree.
    """
    paths = {"--reference": reference, "--training-sets": training_sets}
    paths = {k: v for k, v in paths.items() if v is not None}
    paths["--artifacts"] = artifacts
    try:
        return model_from_names(paths)
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def _check_artifacts(reference, training_sets, artifacts: pathlib.Path) -> bool:
    """Validate ``--artifacts``; return True when it is an add-training run."""
    if reference is None and training_sets is None:
        raise CliError("give --reference, --training-sets, or both.")
    if not artifacts.exists():
        return False
    if reference is not None:
        raise CliError(
            f"artifacts folder '{artifacts}' already exists; delete or move it, or "
            "give only --training-sets to add training sets to it."
        )
    if not (artifacts / REFERENCE_DIR).is_dir() or (artifacts / TRAINING_DIR).exists():
        raise CliError(
            f"cannot add training sets to '{artifacts}': training sets can only be "
            f"added to artifacts that have {REFERENCE_DIR}/ and no {TRAINING_DIR}/."
        )
    return True


def _read_reference(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    logger.info(f"fit | reading reference CSV {path}")
    import pandas as pd

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise CliError(f"could not read reference CSV '{path}': {exc}") from exc


@click.command(
    "fit",
    help=(
        "Fit the quality scores of one model and save them in an artifacts folder."
        "\n\nNames must carry the model, e.g. reference_eos4e40_v1.csv."
    ),
    short_help="Fit quality scores and save artifacts.",
)
@click.option(
    "--reference",
    "-r",
    metavar="CSV",
    help="The model's predictions on the reference library.",
)
@click.option(
    "--training-sets",
    "-t",
    metavar="DIR",
    help="Folder of training sets, one CSV per output column.",
)
@click.option(
    "--artifacts", "-a", required=True, metavar="DIR", help="Artifacts folder to write."
)
@click.option(
    "--exclude",
    multiple=True,
    metavar="SCORES",
    help=("Scores not to fit, comma-separated, e.g. ref_signal."),
)
@verbose_option
def fit(
    reference: str | None,
    training_sets: str | None,
    artifacts: str,
    exclude: tuple[str, ...],
    verbose: bool,
) -> None:
    """Fit the reference and/or training scores and save the artifacts.

    Parameters
    ----------
    reference : str or None
        Reference predictions CSV.
    training_sets : str or None
        Training-sets folder.
    artifacts : str
        Artifacts folder (new, or existing to add training sets to).
    exclude : tuple of str
        Scores not to fit.
    verbose : bool
        Also print debug messages and full tracebacks.
    """
    options = dict(locals())
    run_command(lambda: _fit(**options), verbose=verbose, command="fit")


def _fit(*, reference, training_sets, artifacts, exclude, verbose) -> None:
    started = time.perf_counter()
    folder = pathlib.Path(artifacts)
    add_training = _check_artifacts(reference, training_sets, folder)
    eos_id, version = resolve_model(reference, training_sets, artifacts)
    excluded = parse_exclude(exclude)
    fitted = [
        s
        for s in (REFERENCE_SCORES if reference else ())
        + (TRAINING_SCORES if training_sets else ())
        if s not in excluded
    ]
    console.summary_panel(
        "eosquality · fit" + (" (add training sets)" if add_training else ""),
        [
            ("model", f"{eos_id} {version}"),
            ("reference", console.path(reference) if reference else "—"),
            ("training sets", console.path(training_sets) if training_sets else "—"),
            ("scores", ", ".join(fitted) or "—"),
            ("artifacts", console.path(folder)),
        ],
        icon="◆",
    )
    from eosquality.quality import ErsiliaQuality

    if add_training:
        with logger.log_file(folder / LOG_FILE) as log_path:
            logger.info(f"fit | adding training sets {training_sets} → {folder}")
            eq = ErsiliaQuality.add_training(
                folder, training_sets, eos_id=eos_id, version=version, exclude=excluded
            )
    else:
        with staged_log(folder / LOG_FILE) as log_path:
            logger.info(f"fit | eosquality {eos_id} {version} → {folder}")
            eq = ErsiliaQuality(verbose=verbose).fit(
                _read_reference(reference),
                training_sets,
                eos_id=eos_id,
                version=version,
                exclude=excluded,
            )
            with console.section("Save") as section:
                eq.save(folder)
                section.summary = f"{console.folder_size(folder)}"
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
            ("scores", ", ".join(SCORE_NAMES[s] for s in scores)),
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
