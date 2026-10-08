"""``eosquality run`` — score query molecules against saved artifacts."""

from __future__ import annotations

import json
import pathlib
import time
from typing import TYPE_CHECKING

import click

from eosquality.cli._common import (
    CliError,
    jobs_option,
    require_new_path,
    run_command,
    verbose_option,
)
from eosquality.exceptions import IncompatibleArtifactsError
from eosquality.utils import console
from eosquality.utils.identifiers import model_from_names
from eosquality.utils.logging import logger

if TYPE_CHECKING:  # heavy imports happen inside the command, not at CLI start-up
    from eosquality.quality import ErsiliaQuality


def sibling_path(output: str, suffix: str) -> pathlib.Path:
    """A file next to the scores CSV, named after its stem.

    Parameters
    ----------
    output : str
        The scores CSV path, e.g. ``scores.csv``.
    suffix : str
        What follows the stem, e.g. ``.training_details.csv``.

    Returns
    -------
    pathlib.Path
        ``scores.training_details.csv`` next to ``scores.csv``.
    """
    path = pathlib.Path(output)
    return path.with_name(path.stem + suffix)


def _load_artifacts(path: str) -> ErsiliaQuality:
    from eosquality.quality import ErsiliaQuality

    try:
        return ErsiliaQuality.load(path)
    except FileNotFoundError as exc:
        raise CliError(
            f"artifact at '{path}' is incomplete — refit may be required: {exc}"
        ) from exc
    except IncompatibleArtifactsError as exc:
        raise CliError(
            f"artifact at '{path}' is not compatible with this eosquality install: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CliError(
            f"artifact at '{path}' has a malformed JSON file: {exc}"
        ) from exc


@click.command(
    "run",
    help=(
        "Score query molecules with every score in the artifacts."
        "\n\nThe artifacts and output names must carry the model, e.g. "
        "quality_eos4e40_v1.csv."
    ),
    short_help="Score query data against fitted artifacts.",
)
@click.option(
    "--input", "-i", "input_path", required=True, metavar="PATH", help="Query CSV file."
)
@click.option(
    "--artifacts",
    "-a",
    required=True,
    metavar="PATH",
    help="Artifacts folder from 'fit'.",
)
@click.option(
    "--output", "-o", required=True, metavar="CSV", help="Scores CSV to write (.csv)."
)
@click.option(
    "--details",
    is_flag=True,
    help=(
        "Also write <output stem>.reference_details.csv and "
        ".training_details.csv (per-column values, nearest training molecules)."
    ),
)
@jobs_option
@verbose_option
def run(
    input_path: str,
    artifacts: str,
    output: str,
    details: bool,
    jobs: int,
    verbose: bool,
) -> None:
    """Score a query CSV and write the scores CSV (and, with ``details``, the details CSVs).

    Parameters
    ----------
    input_path : str
        Query CSV.
    artifacts : str
        Fitted artifacts folder.
    output : str
        Scores CSV path (must not exist).
    details : bool
        Also write the reference and training details CSVs.
    jobs : int
        Worker processes for the descriptors.
    verbose : bool
        Print debug messages and diagnostic tables.
    """
    run_command(
        lambda: _run(input_path, artifacts, output, details),
        verbose=verbose,
        command="run",
        jobs=jobs,
    )


def _read_query(input_path: str):
    """The query CSV, or a :class:`CliError` when it cannot be read or is empty."""
    import pandas as pd

    try:
        query = pd.read_csv(input_path)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise CliError(f"could not read query CSV '{input_path}': {exc}") from exc
    if query.empty:
        raise CliError(f"query CSV '{input_path}' has no rows.")
    return query


def _check_details_paths(eq, details_path, reference_details) -> None:
    """Refuse to overwrite a details file this run would write."""
    if "training" in eq.modalities_:
        require_new_path(details_path, "training details path")
    if eq.typicality is not None or eq.extremity is not None:
        require_new_path(reference_details, "reference details path")


def _run(input_path, artifacts, output, details) -> None:
    """Body of ``eosquality run`` (see :func:`run`)."""
    try:
        named = model_from_names({"--artifacts": artifacts, "--output": output})
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    if pathlib.Path(output).suffix.lower() != ".csv":
        raise CliError(f"--output must be a .csv file (got '{output}').")
    if not pathlib.Path(artifacts).is_dir():
        raise CliError(f"artifacts folder '{artifacts}' does not exist.")
    require_new_path(output)
    started = time.perf_counter()
    details_path = sibling_path(output, ".training_details.csv")
    reference_details = sibling_path(output, ".reference_details.csv")
    log_path = sibling_path(output, ".log")
    with logger.log_file(log_path):
        logger.info(f"run | {input_path} against {artifacts} → {output}")
        query = _read_query(input_path)
        eq = _load_artifacts(artifacts)
        eos_id, version = eq._model_id()
        if (eos_id, version) != named:
            raise CliError(
                f"the names say {named[0]} {named[1]}, but the artifacts in "
                f"'{artifacts}' were fitted for {eos_id} {version}."
            )
        if details:  # fail before the scoring work
            _check_details_paths(eq, details_path, reference_details)
        console.summary_panel(
            "eosquality · run",
            [
                ("model", f"{eos_id} {version}"),
                ("modalities", " + ".join(eq.modalities_)),
                ("query", f"{console.path(input_path)}  [dim]{len(query):,} rows[/]"),
                ("artifacts", console.path(artifacts)),
                ("output", console.path(output)),
            ],
            icon="◆",
        )
        result = eq.run(query)
        _write_outputs(query, result, output, details_path, reference_details, details)
    rows = [("queries", f"{len(query):,}"), ("scores", console.path(output))]
    if details and result.training_details is not None:
        rows.append(("training details", console.path(details_path)))
    if details and result.reference_details is not None:
        rows.append(("reference details", console.path(reference_details)))
    rows += [
        ("log", console.path(log_path)),
        ("time", console.elapsed(time.perf_counter() - started)),
    ]
    console.summary_panel("Run complete", rows, color="green", icon="✓")


def _write_outputs(
    query, result, output, details_path, reference_details, details
) -> None:
    """Write the scores CSV (with ``key``/``input``) and, if asked, the details CSVs."""
    import pandas as pd

    with console.section("Write outputs") as section:
        prepend = [c for c in ("key", "input", "smiles") if c in query.columns]
        pd.concat(
            [
                query[prepend].reset_index(drop=True),
                result.scores.reset_index(drop=True),
            ],
            axis=1,
        ).to_csv(output, index=False)
        console.success(f"scores → {console.path(output)}")
        if details and result.training_details is not None:
            result.training_details.to_csv(details_path, index=False)
            console.success(f"training details → {console.path(details_path)}")
        if details and result.reference_details is not None:
            result.reference_details.to_csv(reference_details, index=False)
            console.success(f"reference details → {console.path(reference_details)}")
        section.summary = console.plural(len(result.scores.columns), "column")
