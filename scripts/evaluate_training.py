"""Method check for training-modality scores (Protocol B).

Given one training set (``smiles``, ``y``), split it 80/20 by Murcko
scaffold, fit eosquality's training modality on the 80% and a surrogate
model (random forest on Morgan bits) on the same 80%, then ask whether the
training scores of the held-out 20% predict the surrogate's error there.
Without an external labelled set this validates the *method*, not the
Ersilia model itself.

Reported for each training score (training_distance, training_distance_raw,
training_difficulty), with 95% bootstrap intervals where noted:
- Spearman(score, |error|), with CI: > 0 means a higher score ↔ larger error;
  0 is the random baseline.
- AUROC for flagging the top-quartile errors, with CI (random: 0.5).
- Sparsification gain: how much of the oracle's error reduction (dropping the
  truly worst molecules first) is recovered by dropping the highest-scored
  molecules first (random: 0, oracle: 1).
- Mean |error| per score quartile.
- UNIQUE's ranking-based metrics (Novartis UNIQUE, ``evaluation_metrics.py``),
  on cumulative bins of the data ordered by score (``nbins = min(10, n/5)``)
  with MAE as the bin performance: AUC difference to the oracle ordering
  (0 is perfect), performance drop (MAE of all data and of the highest-score
  bin, each over the MAE of the lowest-score bin; > 1 is good), and the
  increasing/decreasing coefficients (fraction of consecutive bins where MAE
  goes the expected way; 1 is perfect).

    python scripts/evaluate_training.py --csv train.csv [--y-col y] [--max-n 20000]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

from eosquality import ErsiliaQuality
from eosquality.training.folds import morgan_bits

RDLogger.DisableLog("rdApp.*")
SEED = 0


def scaffold_split(smiles: list[str], test_frac: float = 0.2) -> np.ndarray:
    """Boolean test mask: whole scaffold groups, random order, until ~test_frac.

    Parameters
    ----------
    smiles : list of str
        Molecules to split.
    test_frac : float, optional
        Target fraction of molecules in the test split.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` for test molecules.
    """
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles]
    groups = pd.Series(range(len(smiles))).groupby(scaffolds).apply(list).tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    test = np.zeros(len(smiles), dtype=bool)
    for g in groups:
        if test.sum() >= test_frac * len(smiles):
            break
        test[g] = True
    return test


def bootstrap(fn, *arrays, n=500):
    """95% bootstrap interval of a statistic over paired arrays.

    Parameters
    ----------
    fn : callable
        Statistic of the resampled arrays.
    *arrays : numpy.ndarray
        Paired arrays, resampled together.
    n : int, optional
        Bootstrap replicates.

    Returns
    -------
    numpy.ndarray
        ``[lower, upper]``.
    """
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(arrays[0]), len(arrays[0]))
        try:
            stats.append(fn(*(a[i] for a in arrays)))
        except ValueError:
            continue
    return np.percentile(stats, [2.5, 97.5])


SCORES = ("training_distance", "training_distance_raw", "training_difficulty")


def sparsification_gain(score: np.ndarray, err: np.ndarray) -> float:
    """Share of the oracle's sparsification area recovered by ``score``.

    Molecules are dropped from the highest score down; after each drop the
    mean error of those kept is recorded. The area between the random curve
    (constant mean error) and the score's curve is divided by the same area
    for the oracle, which drops by true error.

    Parameters
    ----------
    score : numpy.ndarray
        Higher means expected to be worse.
    err : numpy.ndarray
        Absolute errors.

    Returns
    -------
    float
        0 for random ranking, 1 for the oracle.
    """

    def curve(order: np.ndarray) -> np.ndarray:
        kept = err[order][::-1]  # ascending score: the kept part is a prefix
        return np.cumsum(kept)[::-1] / np.arange(len(err), 0, -1)

    random_area = err.mean() * len(err)
    score_area = curve(np.argsort(score)).sum()
    oracle_area = curve(np.argsort(err)).sum()
    return float((random_area - score_area) / (random_area - oracle_area))


def _cumulative_mae(order: np.ndarray, err: np.ndarray, nbins: int) -> np.ndarray:
    """MAE of the first 1/nbins, 2/nbins, … of the rows taken in ``order``."""
    sizes = (np.linspace(0, 1, nbins + 1)[1:] * len(err)).astype(int)
    return np.array([err[order[:s]].mean() for s in sizes])


def unique_ranking_metrics(score: np.ndarray, err: np.ndarray) -> dict:
    """UNIQUE's ranking-based evaluation metrics for one score.

    Mirrors ``unique.evaluation.evaluation_metrics`` (binning as in
    ``get_indices_bin``, MAE as the per-bin performance). The two
    "coefficients" are computed as described there: UNIQUE's own code uses
    ``<`` for both and sums bin indices instead of counting bins, which this
    does not copy.

    Parameters
    ----------
    score : numpy.ndarray
        Higher means expected to be worse.
    err : numpy.ndarray
        Absolute errors.

    Returns
    -------
    dict
        ``auc_difference``, ``performance_drop_all_vs_low``,
        ``performance_drop_high_vs_low``, ``increasing_coefficient`` and
        ``decreasing_coefficient``.
    """
    nbins = max(2, min(10, len(err) // 5))
    x = np.linspace(0, 1, nbins + 1)[1:]
    increasing = _cumulative_mae(np.argsort(score, kind="stable"), err, nbins)
    decreasing = _cumulative_mae(np.argsort(-score, kind="stable"), err, nbins)
    best = _cumulative_mae(np.argsort(err, kind="stable"), err, nbins)
    return {
        "auc_difference": float(np.trapezoid(increasing, x) - np.trapezoid(best, x)),
        "performance_drop_all_vs_low": float(increasing[-1] / increasing[0]),
        "performance_drop_high_vs_low": float(decreasing[0] / increasing[0]),
        "increasing_coefficient": float(np.mean(np.diff(increasing) > 0)),
        "decreasing_coefficient": float(np.mean(np.diff(decreasing) < 0)),
    }


def metrics(score: np.ndarray, err: np.ndarray) -> dict:
    """Ranking metrics of one score against the surrogate's absolute errors.

    Parameters
    ----------
    score : numpy.ndarray
        Training score of the held-out molecules (NaN rows are dropped).
    err : numpy.ndarray
        Absolute errors of the held-out molecules.

    Returns
    -------
    dict
        Spearman, AUROC (with intervals), sparsification gain and error by
        score quartile.
    """
    ok = np.isfinite(score)
    score, err = score[ok], err[ok]
    big = err >= np.quantile(err, 0.75)
    quart = pd.qcut(score, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    return {
        "n": int(ok.sum()),
        "spearman": spearmanr(score, err)[0],
        "spearman_ci": bootstrap(lambda d, e: spearmanr(d, e)[0], score, err),
        "auroc": roc_auc_score(big, score),
        "auroc_ci": bootstrap(lambda b, d: roc_auc_score(b, d), big, score),
        "sparsification_gain": sparsification_gain(score, err),
        "unique": unique_ranking_metrics(score, err),
        "error_by_quartile": pd.Series(err)
        .groupby(quart, observed=True)
        .mean()
        .round(4)
        .to_dict(),
    }


def evaluate(df: pd.DataFrame, name: str = "column") -> dict:
    """Run the scaffold-split method check on one training set.

    Parameters
    ----------
    df : pandas.DataFrame
        ``smiles`` and ``y`` columns.
    name : str, optional
        Column name used for the temporary training file.

    Returns
    -------
    dict
        Sizes, label kind and :func:`metrics` for each training score.
    """
    df = df.dropna(subset=["smiles", "y"]).reset_index(drop=True)
    df = df[df.smiles.map(lambda s: Chem.MolFromSmiles(s) is not None)].reset_index(
        drop=True
    )
    test = scaffold_split(df.smiles.tolist())
    train_df, test_df = df[~test], df[test]
    binary = set(np.unique(df.y)) <= {0.0, 1.0}

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "training_eos0aaa_v1"
        folder.mkdir()
        train_df[["smiles", "y"]].to_csv(folder / f"{name}.csv", index=False)
        eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=folder)
        q = pd.DataFrame(
            {"key": [str(i) for i in test_df.index], "input": test_df.smiles}
        )
        scores = eq.run(q).scores

    X_tr, X_te = (
        morgan_bits(train_df.smiles.tolist()),
        morgan_bits(test_df.smiles.tolist()),
    )
    if binary:
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
        model.fit(X_tr, train_df.y)
        err = np.abs(model.predict_proba(X_te)[:, 1] - test_df.y.to_numpy())
    else:
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
        model.fit(X_tr, train_df.y)
        err = np.abs(model.predict(X_te) - test_df.y.to_numpy())
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "binary": binary,
        "scores": {
            c: metrics(scores[c].to_numpy(), err) for c in SCORES if c in scores
        },
    }


def report(name: str, r: dict) -> None:
    """Print one evaluation result.

    Parameters
    ----------
    name : str
        Dataset name.
    r : dict
        Output of :func:`evaluate`.
    """
    print(
        f"\n== {name} | n_train={r['n_train']:,} n_test={r['n_test']:,} | "
        f"{'binary' if r['binary'] else 'continuous'} y"
    )
    for score, m in r["scores"].items():
        lo, hi = m["spearman_ci"]
        alo, ahi = m["auroc_ci"]
        print(f"-- {score} (n={m['n']:,})")
        print(
            f"   Spearman(score, |error|) = {m['spearman']:.3f}  [{lo:.3f}, {hi:.3f}]"
        )
        print(f"   AUROC top-quartile error = {m['auroc']:.3f}  [{alo:.3f}, {ahi:.3f}]")
        print(f"   sparsification gain      = {m['sparsification_gain']:.3f}")
        print(f"   mean |error| by quartile: {m['error_by_quartile']}")
        u = m["unique"]
        print(
            "   UNIQUE: AUC difference = "
            f"{u['auc_difference']:.3f} | performance drop all/low = "
            f"{u['performance_drop_all_vs_low']:.2f}, high/low = "
            f"{u['performance_drop_high_vs_low']:.2f} | increasing/decreasing "
            f"coefficient = {u['increasing_coefficient']:.2f}/"
            f"{u['decreasing_coefficient']:.2f}"
        )


def main() -> None:
    """Command-line entry point (see the module docstring for usage)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--y-col", default="y")
    parser.add_argument(
        "--max-n", type=int, default=20_000, help="random subsample (default 20,000)"
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv).rename(
        columns={args.smiles_col: "smiles", args.y_col: "y"}
    )
    if len(df) > args.max_n:
        df = df.sample(n=args.max_n, random_state=SEED)
    report(Path(args.csv).stem, evaluate(df[["smiles", "y"]], name=args.y_col))


if __name__ == "__main__":
    main()
