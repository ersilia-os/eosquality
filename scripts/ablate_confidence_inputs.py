"""Is `trn_difficulty` just the surrogate's confidence?

The error model sees four scalars: ``nn1_tanimoto``, ``nn5_tanimoto``,
``ensemble_variance`` and ``surrogate_score``. The last two are views of how
confident the surrogate is, and for a binary label the absolute error
``|y - p|`` is largely determined by ``p`` itself. So the score could be
reporting "the surrogate is unsure" rather than "this molecule sits in
sparse chemistry".

This script measures that: it evaluates the training scores as usual
(`evaluate_training.py`, Protocol B), then re-evaluates with an error model
that sees only the two neighbour similarities, and prints the difference.
The result is quoted in ``docs/status.md``.

    python scripts/ablate_confidence_inputs.py [--endpoints eos7m30:herg,...]
        [--max-n 6000]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_training as ev  # noqa: E402

from eosquality.scores import _error_model as em  # noqa: E402

TRAINING_DIR = REPO / "data" / "training_examples"
DEFAULT_ENDPOINTS = ["eos7m30:herg", "eos7m30:bbb_martins", "eos4e40:inhibition_50um"]
# Stand-ins that differ from the surrogate (the transfer case).
BLACK_BOXES = ("xgb_physchem", "knn_morgan")


def distance_only_inputs(*, dist, prediction, variance, binary=False) -> np.ndarray:
    """The two neighbour similarities, with both confidence views removed."""
    return np.column_stack([1.0 - dist[:, 0], 1.0 - dist.mean(axis=1)]).astype(
        np.float64
    )


def distance_only_names(binary: bool = False) -> list[str]:
    """Input names matching :func:`distance_only_inputs`.

    Parameters
    ----------
    binary : bool, optional
        Unused; kept for the signature the error model expects.

    Returns
    -------
    list of str
    """
    return ["nn1_tanimoto", "nn5_tanimoto"]


def mean_difficulty_spearman(eos: str, column: str, max_n: int) -> float:
    """Spearman of ``trn_difficulty`` vs held-out |error|, averaged over boxes.

    Parameters
    ----------
    eos : str
        Model id.
    column : str
        Output column (the training file's stem).
    max_n : int
        Random subsample of the training set.

    Returns
    -------
    float
    """
    path = TRAINING_DIR / f"training_{eos}_v1" / f"{column}.csv"
    df = pd.read_csv(path)
    label = next(c for c in ("y", "value") if c in df.columns)
    df = df.rename(columns={label: "y"})
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=ev.SEED)
    result = ev.evaluate(df[["smiles", "y"]], name=column, black_boxes=BLACK_BOXES)
    values = [
        m["spearman"]
        for per_score in result["black_boxes"].values()
        for score, m in per_score.items()
        if score == "trn_difficulty"
    ]
    return float(np.mean(values))


def main() -> None:
    """Command-line entry point (see the module docstring for usage)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
    parser.add_argument("--max-n", type=int, default=6000)
    args = parser.parse_args()
    endpoints = [item.split(":") for item in args.endpoints.split(",")]

    baseline = {}
    for eos, column in endpoints:
        baseline[column] = mean_difficulty_spearman(eos, column, args.max_n)
        print(f"all four      {column:24s} {baseline[column]:.3f}", flush=True)

    em._inputs, em.feature_names = distance_only_inputs, distance_only_names
    ablated = {}
    for eos, column in endpoints:
        ablated[column] = mean_difficulty_spearman(eos, column, args.max_n)
        print(f"distances only{column:24s} {ablated[column]:.3f}", flush=True)

    print("\nendpoint, all four, distances only, cost of the ablation")
    for column, value in baseline.items():
        print(
            f"{column}, {value:.3f}, {ablated[column]:.3f}, {value - ablated[column]:+.3f}"
        )


if __name__ == "__main__":
    main()
