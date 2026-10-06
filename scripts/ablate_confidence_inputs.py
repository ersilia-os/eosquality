"""Is `trn_difficulty`'s advantage on binary endpoints just classifier confidence?

For a binary label the absolute error ``|y − p|`` is nearly a function of the
surrogate's confidence, and the error model sees that confidence three ways:
``probability_top1``, the ensemble variance, and the prediction itself (for a
binary label the prediction *is* P(y = 1)). If the high binary scores came
from rediscovering that identity, removing all three should collapse them.

This script measures it: it evaluates the training scores as usual
(`evaluate_training.py`, Protocol B), then re-evaluates with an error model
that sees only structure — MACCS keys, the kNN distance and the three KDE
log-densities — and prints the difference. The result is quoted in
``docs/status.md``.

    python scripts/ablate_confidence_inputs.py [--endpoints eos7m30:herg,…]
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


def structure_only_inputs(
    *, dist, prediction, variance, log_density, maccs, binary
) -> np.ndarray:
    """Feature set (i) with every view of the surrogate's confidence removed."""
    return np.hstack([maccs, dist.mean(axis=1)[:, None], log_density]).astype(
        np.float64
    )


def structure_only_names(binary: bool) -> list[str]:
    """Input names matching :func:`structure_only_inputs`.

    Parameters
    ----------
    binary : bool
        Unused; kept for the signature the error model expects.

    Returns
    -------
    list of str
    """
    return [*em.MACCS_NAMES, "knn_distance", *em.KDE_NAMES]


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
        print(f"feature set (i)   {column:24s} {baseline[column]:.3f}", flush=True)

    em._inputs, em.feature_names = structure_only_inputs, structure_only_names
    ablated = {}
    for eos, column in endpoints:
        ablated[column] = mean_difficulty_spearman(eos, column, args.max_n)
        print(f"structure only    {column:24s} {ablated[column]:.3f}", flush=True)

    print("\nendpoint, feature set (i), structure only, cost of the ablation")
    for column, value in baseline.items():
        print(
            f"{column}, {value:.3f}, {ablated[column]:.3f}, {value - ablated[column]:+.3f}"
        )


if __name__ == "__main__":
    main()
