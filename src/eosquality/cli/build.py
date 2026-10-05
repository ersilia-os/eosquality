"""``eosquality build`` — build a vector index from a SMILES CSV.

Release / maintenance tool. End users do not normally call this; the
canonical reference library ships with each release. Used to prepare a
replacement library for the next release, or to build a non-canonical
index for internal testing (pass the result to ``fit --vector-index``).
Writes the Morgan FP index plus the physchem and MACCS descriptor matrices.
"""

import pathlib
import time

import click
import pandas as pd

from eosquality.basic_descriptors import BasicDescriptors
from eosquality.cli._common import CliError, run_command, verbose_option
from eosquality.utils import console
from eosquality.vectorindex import VectorIndex


@click.command(
    "build",
    help=(
        "Build a Morgan-fingerprint kNN index (plus physchem and MACCS matrices) "
        "for a SMILES library. A release / maintenance tool: use it to prepare "
        "the next canonical reference library, or a non-canonical index for "
        "testing (pass the result to 'fit --vector-index')."
    ),
    short_help="Build a reference-library vector index from a SMILES CSV.",
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
    "--output", "-o", required=True, metavar="PATH", help="Output folder for the index."
)
@click.option(
    "--max-k",
    default=50,
    show_default=True,
    metavar="K",
    help="Maximum k to pre-compute for self-kNN.",
)
@click.option(
    "--radius", default=2, show_default=True, metavar="R", help="Morgan radius."
)
@click.option(
    "--n-bits", default=2048, show_default=True, metavar="N", help="Morgan bits."
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    metavar="N",
    help="Truncate input to the first N molecules (for testing).",
)
@verbose_option
def build(
    input_path: str,
    output: str,
    max_k: int,
    radius: int,
    n_bits: int,
    max_samples: int | None,
    verbose: bool,
) -> None:
    """Build the vector index and descriptor matrices for a SMILES library.

    Parameters
    ----------
    input_path : str
        Library CSV with a ``smiles`` column.
    output : str
        Output folder.
    max_k, radius, n_bits : int
        Self-kNN depth and Morgan fingerprint parameters.
    max_samples : int or None
        Optional truncation for testing.
    verbose : bool
        Print debug messages.
    """

    run_command(
        lambda: _build(input_path, output, max_k, radius, n_bits, max_samples, verbose),
        verbose=verbose,
        command="build",
    )


def _read_library(input_path: str, max_samples: int | None) -> list[str]:
    """SMILES of the library CSV, optionally truncated."""
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


def _build(input_path, output, max_k, radius, n_bits, max_samples, verbose) -> None:
    """Body of ``eosquality build`` (see :func:`build`)."""
    smiles = _read_library(input_path, max_samples)
    started = time.perf_counter()
    console.summary_panel(
        "eosquality · build",
        [
            (
                "library",
                f"{console.path(input_path)}  [dim]{len(smiles):,} molecules[/]",
            ),
            ("fingerprint", f"Morgan r={radius} · {n_bits} bits · max_k={max_k}"),
            ("output", console.path(output)),
        ],
        icon="◆",
    )
    steps = console.Steps(3)
    with console.section("Build") as section:
        try:
            with steps("Vector index and self-kNN") as st:
                VectorIndex.build(
                    smiles=smiles,
                    output_dir=output,
                    max_k=max_k,
                    radius=radius,
                    n_bits=n_bits,
                    verbose=verbose,
                    library_name=pathlib.Path(input_path).stem,
                )
                st.summary = f"{len(smiles):,} molecules"
            with steps("Physicochemical descriptors"):
                BasicDescriptors.build_physchem(smiles, output)
            with steps("MACCS keys"):
                BasicDescriptors.build_maccs(smiles, output)
        except Exception as exc:
            raise CliError(f"index build failed: {exc}") from exc
        section.summary = console.folder_size(output)
    console.summary_panel(
        "Build complete",
        [
            ("index", console.path(output)),
            ("time", console.elapsed(time.perf_counter() - started)),
        ],
        color="green",
        icon="✓",
    )
