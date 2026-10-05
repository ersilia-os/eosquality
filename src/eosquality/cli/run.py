"""``eosquality run`` — score query molecules against saved artifacts."""

import json
import pathlib

import click
import pandas as pd

from eosquality.cli._common import (
    CliError,
    require_new_path,
    run_command,
    say,
    verbose_option,
)
from eosquality.exceptions import IncompatibleArtifactsError
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


def _load_artifacts(path: str) -> ErsiliaQuality:
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
        "Score query molecules with every score in the artifacts. The output CSV "
        "has the query's 'key' and 'input' columns, then each fitted score with "
        "its '*_raw' companion. If the artifacts hold a training modality, a "
        "second CSV with one row per (query, output column) and the nearest "
        "training molecules is written next to it."
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
    help="Artifacts folder produced by 'eosquality fit'.",
)
@click.option(
    "--output", "-o", required=True, metavar="PATH", help="Scores CSV to write."
)
@click.option(
    "--training-details",
    default=None,
    metavar="PATH",
    help=(
        "Per-column training details CSV (default: <output stem>"
        ".training_details.csv). Only written with a training modality."
    ),
)
@verbose_option
def run(
    input_path: str,
    artifacts: str,
    output: str,
    training_details: str | None,
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
    training_details : str or None
        Training-details CSV path; defaults next to ``output``.
    verbose : bool
        Print debug messages and diagnostic tables.
    """

    def work():
        if not pathlib.Path(artifacts).is_dir():
            raise CliError(f"artifacts folder '{artifacts}' does not exist.")
        require_new_path(output)
        details_path = training_details or default_details_path(output)
        say(f"→ reading query CSV: {input_path}")
        try:
            query = pd.read_csv(input_path)
        except Exception as exc:
            raise CliError(f"could not read query CSV '{input_path}': {exc}") from exc
        result = _load_artifacts(artifacts).run(query)
        if result.training_details is not None:
            require_new_path(details_path, "training details path")
        prepend = [c for c in ("key", "input") if c in query.columns]
        pd.concat(
            [
                query[prepend].reset_index(drop=True),
                result.scores.reset_index(drop=True),
            ],
            axis=1,
        ).to_csv(output, index=False)
        say(f"→ scores written: {output}")
        if result.training_details is not None:
            result.training_details.to_csv(details_path, index=False)
            say(f"→ training details written: {details_path}")
        if not verbose:
            say(f"Scores saved → {output}", err=False)

    run_command(work, verbose=verbose)
