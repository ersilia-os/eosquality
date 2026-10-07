"""``eosquality fit`` — fit the reference and/or training scores of one model."""

from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING

import click

from eosquality._registry import (
    ALL_SCORES,
    ALSO_EMITS,
    REFERENCE_SCORES,
    SCORE_NAMES,
    TRAINING_SCORES,
)
from eosquality.cli._common import (
    CliError,
    require_new_path,
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
        New artifacts folder.
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
    if reference is None and training_sets is None:
        raise CliError("give --reference, --training-sets, or both.")
    require_new_path(artifacts, "artifacts folder")
    eos_id, version = resolve_model(reference, training_sets, artifacts)
    excluded = parse_exclude(exclude)
    fitted = [
        s
        for s in (REFERENCE_SCORES if reference else ())
        + (TRAINING_SCORES if training_sets else ())
        if s not in excluded
    ]
    console.summary_panel(
        "eosquality · fit",
        [
            ("model", f"{eos_id} {version}"),
            ("reference", console.path(reference) if reference else "—"),
            ("training sets", console.path(training_sets) if training_sets else "—"),
            ("scores", _score_list(fitted)),
            ("artifacts", console.path(folder)),
        ],
        icon="◆",
    )
    from eosquality.quality import ErsiliaQuality

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


def _score_list(names) -> str:
    """Comma-separated score names, with the extra column a score also writes."""
    shown = []
    for name in names:
        shown += [name, *ALSO_EMITS.get(name, ())]
    return ", ".join(shown) or "—"


def _fit_summary(eq, folder: pathlib.Path, log_path: pathlib.Path, started) -> None:
    """Final ``✓ Fit complete`` panel."""
    eos_id, version = eq._model_id()
    scores = list(eq._components()) + list(eq._training_components())
    console.summary_panel(
        "Fit complete",
        [
            ("model", f"{eos_id} {version}"),
            ("modalities", " + ".join(eq.modalities_)),
            ("scores", _score_list(SCORE_NAMES[s] for s in scores)),
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
