"""``eosquality build`` — build a vector index from a SMILES CSV.

Release / maintenance tool. End users do not normally call this; the
canonical reference library ships with each release. Used to prepare a
replacement library for the next release, or to build a non-canonical
index for internal testing (pass the result to ``fit --vector-index``).
Writes the Morgan FP index plus the physchem and MACCS descriptor matrices.
"""

import pathlib

import click
import pandas as pd

from eosquality.basic_descriptors import BasicDescriptors
from eosquality.cli._common import CliError, run_command, say, verbose_option
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

    def _work():
        try:
            df = pd.read_csv(input_path)
        except Exception as exc:
            raise CliError(
                f"could not read library file '{input_path}': {exc}"
            ) from exc
        if "smiles" not in df.columns:
            raise CliError(
                f"library CSV must contain a 'smiles' column (found: {list(df.columns)})"
            )
        smiles = list(df["smiles"])[:max_samples] if max_samples else list(df["smiles"])
        say(
            f"Building vector index for {len(smiles):,} molecules → {output}", err=False
        )
        try:
            VectorIndex.build(
                smiles=smiles,
                output_dir=output,
                max_k=max_k,
                radius=radius,
                n_bits=n_bits,
                verbose=verbose,
                library_name=pathlib.Path(input_path).stem,
            )
            BasicDescriptors.build_physchem(smiles, output)
            BasicDescriptors.build_maccs(smiles, output)
        except Exception as exc:
            raise CliError(f"index build failed: {exc}") from exc
        say(f"Vector index saved → {output}", err=False)

    run_command(_work, verbose=verbose)
