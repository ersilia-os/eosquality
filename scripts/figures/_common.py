"""Shared loading + labelling for the eosquality figure scripts.

Every script reads the per-(query set, model) score CSVs written by
``scripts/run_all_scores.sh`` (``scores_<set>_1000_<eos>_v1.csv``) from a
scores folder (default ``output/``) and writes a PNG to ``docs/figures/``.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SCORES_DIR = REPO / "output"
DEFAULT_OUT_DIR = REPO / "docs" / "figures"

SCORES = ["typicality", "extremity", "support", "consistency", "signal"]
QUERY_SETS = ["molecules", "drugs", "np_large", "synthetic", "inert"]
QUERY_SET_LABELS = {
    "molecules": "Library sample",
    "drugs": "Drugs",
    "np_large": "Natural products",
    "synthetic": "Synthetic",
    "inert": "Inert",
}
MODELS = ["eos3b5e", "eos4e40", "eos3804", "eos42ez", "eos7m30"]
MODEL_LABELS = {
    "eos3b5e": "MW",
    "eos4e40": "E. coli",
    "eos3804": "A. baumannii",
    "eos42ez": "Cytotox",
    "eos7m30": "ADMET",
}

_FNAME_RE = re.compile(r"^scores_(.+)_1000_(eos\w+)_v1\.csv$")


def model_label(model: str) -> str:
    """Display label ``"<eos id> (<endpoint>)"`` of a model.

    Parameters
    ----------
    model : str
        EOS model id.

    Returns
    -------
    str
    """
    tag = MODEL_LABELS.get(model)
    return f"{model} ({tag})" if tag else model


def parse_args(description: str) -> argparse.Namespace:
    """Parse the common ``--scores-dir`` / ``--out-dir`` / ``--suffix`` flags.

    Parameters
    ----------
    description : str
        Help text of the script.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scores-dir", type=pathlib.Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--suffix", default="", help="Appended to output filenames.")
    return parser.parse_args()


def load_scores(scores_dir: pathlib.Path) -> pd.DataFrame:
    """Long table of every score CSV, tagged with ``query_set`` and ``model``.

    Parameters
    ----------
    scores_dir : pathlib.Path
        Folder with ``scores_<set>_1000_<eos>_v1.csv`` files.

    Returns
    -------
    pandas.DataFrame
    """
    frames = []
    for csv in sorted(scores_dir.glob("scores_*_v1.csv")):
        m = _FNAME_RE.match(csv.name)
        if not m:
            continue
        df = pd.read_csv(csv)
        # Reference scores are prefixed "ref_" since the CLI redesign; the
        # figures use the short names, so CSVs of either vintage work.
        df = df.rename(columns=lambda c: c[4:] if c.startswith("ref_") else c)
        df = df.rename(columns={"extremity_pct": "extremity"})
        df["query_set"], df["model"] = m.group(1), m.group(2)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No scores_*_1000_<eos>_v1.csv files in {scores_dir}")
    return pd.concat(frames, ignore_index=True)


def present(values: list[str], available) -> list[str]:
    """``values`` restricted to ``available``, plus any extras sorted.

    Parameters
    ----------
    values : list of str
        Preferred order.
    available : iterable of str
        Values actually present.

    Returns
    -------
    list of str
    """
    available = set(available)
    ordered = [v for v in values if v in available]
    return ordered + sorted(available - set(ordered))
