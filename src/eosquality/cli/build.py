"""``eosquality build`` — prepare a reference-library folder from a SMILES CSV.

Release / maintenance tool. End users do not normally call this; the
canonical reference library ships with each release. Used to prepare a
replacement library for the next release, or a non-canonical library for
internal testing (point ``EOSQUALITY_REFERENCE_LIBRARY_PATH`` at the result,
or pass it as ``library=`` to ``ErsiliaQuality.fit``).

The folder holds the library SMILES, a ``metadata.json`` with its identity, and
the connectivity keys that ``ref_match`` / ``ref_scaffold`` look up.
"""

import json
import pathlib
import time
from datetime import UTC, datetime

import click

from eosquality.cli._common import (
    CliError,
    jobs_option,
    require_new_path,
    run_command,
    verbose_option,
)
from eosquality.utils import console


@click.command(
    "build",
    help=("Build a reference-library folder from a SMILES CSV (maintainers)."),
    short_help="Build a reference-library folder from a SMILES CSV.",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    metavar="PATH",
    help="Reference library CSV file (must have a 'smiles' column).",
)
@click.option(
    "--output", "-o", required=True, metavar="PATH", help="Output library folder."
)
@click.option(
    "--name",
    default=None,
    metavar="NAME",
    help=(
        "Library identity (library_name). Default: the CSV file name without its "
        "extension, which must then be a library id such as "
        "ersilia_reference_library_v1."
    ),
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    metavar="N",
    help="Truncate input to the first N molecules (for testing).",
)
@jobs_option
@verbose_option
def build(
    input_path: str,
    output: str,
    name: str | None,
    max_samples: int | None,
    jobs: int,
    verbose: bool,
) -> None:
    """Build the library folder: SMILES, metadata and connectivity keys.

    Parameters
    ----------
    input_path : str
        Library CSV with a ``smiles`` column.
    output : str
        Output folder; it must not exist yet.
    name : str or None
        Library identity; defaults to the CSV file stem, which must be a
        library id (``ersilia_reference_library_vN``).
    max_samples : int or None
        Optional truncation for testing.
    jobs : int
        Worker processes for the standardisation and the connectivity keys.
    verbose : bool
        Print debug messages.
    """

    run_command(
        lambda: _build(input_path, output, name, max_samples),
        verbose=verbose,
        command="build",
        jobs=jobs,
    )


def _read_library(input_path: str, max_samples: int | None) -> list[str]:
    """SMILES of the library CSV, optionally truncated."""
    import pandas as pd

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise CliError(f"could not read library file '{input_path}': {exc}") from exc
    if "smiles" not in df.columns:
        raise CliError(
            f"library CSV must contain a 'smiles' column (found: {list(df.columns)})"
        )
    smiles = list(df["smiles"])
    return smiles[:max_samples] if max_samples else smiles


def _package_version() -> str:
    """Installed eosquality version, or ``"unknown"``."""
    import importlib.metadata

    try:
        return importlib.metadata.version("eosquality")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_library(smiles: list[str], output: str | pathlib.Path, name: str) -> None:
    """Write a library folder: ``smiles.csv``, ``metadata.json``, the match keys.

    The SMILES are standardised (largest fragment, canonical isomeric) before
    their connectivity layers are taken, exactly as a query's are at run time.
    The folder is written whole or not at all: the files are built in a
    temporary folder next to ``output``, which is renamed to ``output`` only
    once every file is complete, so an interrupted build leaves nothing behind.

    Parameters
    ----------
    smiles : list of str
        Library molecules, in order.
    output : str or pathlib.Path
        Folder to write; it must not exist, or be empty.
    name : str
        Library identity, stored as ``library_name``.

    Raises
    ------
    FileExistsError
        If ``output`` exists and is not an empty folder.
    """
    import shutil
    import tempfile

    import numpy as np
    import pandas as pd
    from rdkit import __version__ as rdkit_version

    from eosquality.library.reference import KEYS_FILE, METADATA_FILE, SMILES_FILE
    from eosquality.scores._helpers import _standardize
    from eosquality.scores._match_keys import _layers, save_keys, unique_keys
    from eosquality.utils.parallel import map_rows

    final = pathlib.Path(output)
    if final.exists() and (not final.is_dir() or any(final.iterdir())):
        raise FileExistsError(f"{final} already exists and is not an empty folder.")
    final.parent.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        pd.DataFrame({"smiles": smiles}).to_csv(work / SMILES_FILE, index=False)
        cleaned = np.empty(len(smiles), dtype=object)
        map_rows(_standardize, smiles, cleaned, label="standardise", show_progress=True)
        standardised = [s for s in cleaned if s]
        molecules, scaffolds = _layers(standardised, "library InChIKey layers")
        molecule_keys, scaffold_keys = unique_keys(molecules), unique_keys(scaffolds)
        save_keys(work / KEYS_FILE, molecule_keys, scaffold_keys)
        with open(work / METADATA_FILE, "w") as f:
            json.dump(
                {
                    "n_samples": len(smiles),
                    "n_unparsable": len(smiles) - len(standardised),
                    "n_molecule_keys": int(len(molecule_keys)),
                    "n_scaffold_keys": int(len(scaffold_keys)),
                    "rdkit_version": rdkit_version,
                    "eosquality_version": _package_version(),
                    "build_timestamp": datetime.now(tz=UTC).isoformat(),
                    "library_name": name,
                },
                f,
                indent=2,
            )
        if final.exists():
            final.rmdir()  # empty, checked above
        work.rename(final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def _library_name(input_path: str, name: str | None) -> str:
    """The library identity: ``--name``, else the CSV stem, which must be a library id."""
    from eosquality.library.identity import is_library_id

    if name:
        return name
    stem = pathlib.Path(input_path).stem
    if not is_library_id(stem):
        raise CliError(
            f"the library name '{stem}' (from the CSV file name) is not a library "
            "id like 'ersilia_reference_library_v1'; rename the file or pass --name."
        )
    return stem


def _build(
    input_path: str, output: str, name: str | None, max_samples: int | None
) -> None:
    """Body of ``eosquality build`` (see :func:`build`)."""
    require_new_path(output, "library folder")
    library_name = _library_name(input_path, name)
    smiles = _read_library(input_path, max_samples)
    started = time.perf_counter()
    console.summary_panel(
        "eosquality · build",
        [
            (
                "library",
                f"{console.path(input_path)}  [dim]{len(smiles):,} molecules[/]",
            ),
            ("name", library_name),
            ("output", console.path(output)),
        ],
        icon="◆",
    )
    steps = console.Steps(1)
    with console.section("Build") as section:
        try:
            with steps("Connectivity keys of the molecules and their scaffolds") as st:
                build_library(smiles, output, library_name)
                st.summary = f"{len(smiles):,} molecules"
        except Exception as exc:
            raise CliError(f"library build failed: {exc}") from exc
        section.summary = console.folder_size(output)
    console.summary_panel(
        "Build complete",
        [
            ("library", console.path(output)),
            ("time", console.elapsed(time.perf_counter() - started)),
        ],
        color="green",
        icon="✓",
    )
