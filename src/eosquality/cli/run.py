"""``eosquality run`` — score query molecules against saved artifacts."""

from __future__ import annotations

import json
import pathlib
import time
from typing import TYPE_CHECKING

import click

from eosquality.cli._common import (
    CliError,
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


def default_details_path(output: str) -> str:
    """Path of the training-details CSV next to the scores CSV.

    Parameters
    ----------
    output : str
        The scores CSV path, e.g. ``scores.csv``.

    Returns
    -------
    str
        ``scores.training_details.csv`` next to it.
    """
    p = pathlib.Path(output)
    return str(p.with_name(f"{p.stem}.training_details{p.suffix or '.csv'}"))


def log_path_for(output: str) -> pathlib.Path:
    """Log file of ``run``: ``scores.csv`` → ``scores.log`` (never the output itself).

    Parameters
    ----------
    output : str
        The scores CSV path.

    Returns
    -------
    pathlib.Path
        ``<output stem>.log``, or ``<output>.log`` when the output already
        ends in ``.log``.
    """
    p = pathlib.Path(output)
    log = p.with_suffix(".log")
    return log if log != p else p.with_name(p.name + ".log")


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
        "\n\nNames must carry the model, e.g. query_eos4e40_v1.csv."
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
    "--output", "-o", required=True, metavar="PATH", help="Scores CSV to write."
)
@verbose_option
def run(
    input_path: str,
    artifacts: str,
    output: str,
    verbose: bool,
) -> None:
    """Score a query CSV and write the scores (and training details) CSVs.

    Parameters
    ----------
    input_path : str
        Query CSV.
    artifacts : str
        Fitted artifacts folder.
    output : str
        Scores CSV path (must not exist).
    verbose : bool
        Print debug messages and diagnostic tables.
    """

    run_command(
        lambda: _run(input_path, artifacts, output),
        verbose=verbose,
        command="run",
    )


def _run(input_path, artifacts, output) -> None:
    """Body of ``eosquality run`` (see :func:`run`)."""
    import pandas as pd

    try:
        named = model_from_names(
            {"--input": input_path, "--artifacts": artifacts, "--output": output}
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    if not pathlib.Path(artifacts).is_dir():
        raise CliError(f"artifacts folder '{artifacts}' does not exist.")
    require_new_path(output)
    started = time.perf_counter()
    details_path = default_details_path(output)
    log_path = log_path_for(output)
    with logger.log_file(log_path):
        logger.info(f"run | {input_path} against {artifacts} → {output}")
        try:
            query = pd.read_csv(input_path)
        except Exception as exc:
            raise CliError(f"could not read query CSV '{input_path}': {exc}") from exc
        if query.empty:
            raise CliError(f"query CSV '{input_path}' has no rows.")
        eq = _load_artifacts(artifacts)
        eos_id, version = eq._model_id()
        if (eos_id, version) != named:
            raise CliError(
                f"the names say {named[0]} {named[1]}, but the artifacts in "
                f"'{artifacts}' were fitted for {eos_id} {version}."
            )
        if "training" in eq.modalities_:  # fail before the scoring work
            require_new_path(details_path, "training details path")
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
        _write_outputs(query, result, output, details_path)
    rows = [("queries", f"{len(query):,}"), ("scores", console.path(output))]
    if result.training_details is not None:
        rows.append(("training details", console.path(details_path)))
    rows += [
        ("log", console.path(log_path)),
        ("time", console.elapsed(time.perf_counter() - started)),
    ]
    console.summary_panel("Run complete", rows, color="green", icon="✓")


def _write_outputs(query, result, output, details_path) -> None:
    """Write the scores CSV (with ``key``/``input``) and the training details."""
    import pandas as pd

    with console.section("Write outputs") as section:
        prepend = [c for c in ("key", "input") if c in query.columns]
        pd.concat(
            [
                query[prepend].reset_index(drop=True),
                result.scores.reset_index(drop=True),
            ],
            axis=1,
        ).to_csv(output, index=False)
        console.success(f"scores → {console.path(output)}")
        if result.training_details is not None:
            result.training_details.to_csv(details_path, index=False)
            console.success(f"training details → {console.path(details_path)}")
        section.summary = f"{len(result.scores.columns)} column(s)"
