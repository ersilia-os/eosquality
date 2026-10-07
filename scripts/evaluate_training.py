"""Method check for training-modality scores (Protocol B).

Given one training set (``smiles``, ``y``), split it 80/20 by Murcko
scaffold, fit eosquality's training modality on the 80% and a stand-in
"black-box" model on the same 80%, then ask whether the training scores of
the held-out 20% rank that model's absolute errors there. Without an
external labelled set this validates the *method*, not an Ersilia model.

Black boxes (``--black-box``, default ``all``):
- ``rf_morgan``: random forest on Morgan bits. Same family as the surrogate
  inside training_difficulty, so this case is partly circular.
- ``xgb_physchem``: XGBoost on RDKit physicochemical descriptors.
- ``knn_morgan``: 5-nearest-neighbour model on Morgan bits (Jaccard).
The last two test whether the scores transfer to a model that differs from
the surrogate, which is the situation of an Ersilia black-box model.

Reported for each training score (trn_tanimoto_pct, trn_tanimoto_raw,
trn_difficulty) and black box:
- Spearman(score, |error|) with a 95% bootstrap interval, and the 95th
  percentile of |Spearman| under 1,000 permutations of the score (a value
  below it is indistinguishable from random ranking).
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
- Spearman under data shifts, after Parrondo-Pizarro et al. (JCIM 2026):
  restricted to the 25% and 50% of test molecules most shifted by features
  (distance to the nearest training molecule), labels (distance of y from
  the training median) and discontinuity (Diff5NN on labels: distance of y
  from the mean label of the 5 nearest training molecules).

    python scripts/evaluate_training.py --csv train.csv [--y-col y]
        [--black-box all|rf_morgan|xgb_physchem|knn_morgan] [--max-n 20000]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity, CreateFromBitString
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

from eosquality import ErsiliaQuality
from eosquality.library.physchem import compute_physchem_raw
from eosquality.training.folds import _scaffold, morgan_bits

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
    scaffolds = [_scaffold(s) for s in smiles]
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


SCORES = ("trn_tanimoto_pct", "trn_tanimoto_raw", "trn_difficulty")


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


BLACK_BOXES = ("rf_morgan", "xgb_physchem", "knn_morgan")
SHIFT_FRACTIONS = (0.25, 0.5)


def black_box_errors(name: str, train: pd.DataFrame, test: pd.DataFrame, binary):
    """Absolute held-out errors of one stand-in black-box model.

    Parameters
    ----------
    name : str
        One of ``BLACK_BOXES``.
    train, test : pandas.DataFrame
        ``smiles`` and ``y`` columns.
    binary : bool
        Binary labels (errors are ``|y − P(y = 1)|``).

    Returns
    -------
    numpy.ndarray
        ``(n_test,)`` absolute errors.
    """
    tr, te = train.smiles.tolist(), test.smiles.tolist()
    if name == "xgb_physchem":
        a = compute_physchem_raw(tr, show_progress=False)
        b = compute_physchem_raw(te, show_progress=False)
        kwargs = dict(n_estimators=300, max_depth=6, learning_rate=0.05)
        model = (XGBClassifier if binary else XGBRegressor)(random_state=SEED, **kwargs)
    else:
        a, b = morgan_bits(tr), morgan_bits(te)
        if name == "knn_morgan":
            a, b = a.astype(bool), b.astype(bool)
            model = (KNeighborsClassifier if binary else KNeighborsRegressor)(
                5, metric="jaccard"
            )
        else:
            model = (RandomForestClassifier if binary else RandomForestRegressor)(
                n_estimators=300, n_jobs=-1, random_state=SEED
            )
    y = train.y.to_numpy()
    model.fit(a, y.astype(int) if binary else y)
    predicted = model.predict_proba(b)[:, 1] if binary else model.predict(b)
    return np.abs(test.y.to_numpy() - predicted)


def data_shifts(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-test-molecule feature, label and discontinuity shift.

    Parameters
    ----------
    train, test : pandas.DataFrame
        ``smiles`` and ``y`` columns.

    Returns
    -------
    dict of str to numpy.ndarray
        ``feature`` (1 − Tanimoto to the nearest training molecule),
        ``label`` (``|y − median(y_train)|``) and ``discontinuity``
        (Diff5NN: ``|y − mean y of the 5 nearest training molecules|``).
    """
    fps_tr = [
        CreateFromBitString("".join(map(str, r)))
        for r in morgan_bits(train.smiles.tolist())
    ]
    fps_te = [
        CreateFromBitString("".join(map(str, r)))
        for r in morgan_bits(test.smiles.tolist())
    ]
    y_tr, y_te = train.y.to_numpy(), test.y.to_numpy()
    nearest, diff5 = [], []
    for fp, y in zip(fps_te, y_te, strict=True):
        sims = np.asarray(BulkTanimotoSimilarity(fp, fps_tr))
        top = np.argsort(-sims)[:5]
        nearest.append(1.0 - sims[top[0]])
        diff5.append(abs(y - y_tr[top].mean()))
    return {
        "feature": np.array(nearest),
        "label": np.abs(y_te - np.median(y_tr)),
        "discontinuity": np.array(diff5),
    }


def shift_spearman(score, err, shifts: dict[str, np.ndarray]) -> dict[str, float]:
    """Spearman(score, |error|) on the most-shifted fractions of the test set.

    Parameters
    ----------
    score, err : numpy.ndarray
        Score and absolute error per test molecule.
    shifts : dict of str to numpy.ndarray
        Output of :func:`data_shifts`.

    Returns
    -------
    dict of str to float
        ``"<shift> top <p>%"`` → Spearman on that subset.
    """
    out = {}
    ok = np.isfinite(score)
    for kind, shift in shifts.items():
        order = np.argsort(-shift[ok], kind="stable")
        for frac in SHIFT_FRACTIONS:
            top = order[: max(5, int(frac * ok.sum()))]
            out[f"{kind} top {int(frac * 100)}%"] = float(
                spearmanr(score[ok][top], err[ok][top])[0]
            )
    return out


def permutation_threshold(score, err, n: int = 1000) -> float:
    """95th percentile of |Spearman| when the score is randomly permuted.

    Parameters
    ----------
    score, err : numpy.ndarray
        Score and absolute error per test molecule.
    n : int, optional
        Permutations.

    Returns
    -------
    float
    """
    ok = np.isfinite(score)
    rng = np.random.default_rng(SEED)
    s, e = score[ok], err[ok]
    null = [abs(spearmanr(rng.permutation(s), e)[0]) for _ in range(n)]
    return float(np.percentile(null, 95))


def evaluate(df: pd.DataFrame, name: str = "column", black_boxes=BLACK_BOXES) -> dict:
    """Run the scaffold-split method check on one training set.

    Parameters
    ----------
    df : pandas.DataFrame
        ``smiles`` and ``y`` columns.
    name : str, optional
        Column name used for the temporary training file.
    black_boxes : sequence of str, optional
        Stand-in models whose held-out errors are ranked (``BLACK_BOXES``).

    Returns
    -------
    dict
        Sizes, label kind, and per black box and training score the
        :func:`metrics`, permutation threshold and shift Spearmans.
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
        eq = ErsiliaQuality().fit(
            eos_id="eos0aaa", training_sets=folder, include=["trn_difficulty"]
        )
        q = pd.DataFrame(
            {"key": [str(i) for i in test_df.index], "input": test_df.smiles}
        )
        scores = eq.run(q).scores
    shifts = data_shifts(train_df, test_df)
    results = {}
    for box in black_boxes:
        err = black_box_errors(box, train_df, test_df, binary)
        results[box] = {}
        for c in SCORES:
            if c not in scores:
                continue
            values = scores[c].to_numpy()
            entry = metrics(values, err)
            entry["permutation_95"] = permutation_threshold(values, err)
            entry["shifts"] = shift_spearman(values, err, shifts)
            results[box][c] = entry
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "binary": binary,
        "black_boxes": results,
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
    for box, per_score in r["black_boxes"].items():
        print(f"## black box: {box}")
        for score, m in per_score.items():
            lo, hi = m["spearman_ci"]
            alo, ahi = m["auroc_ci"]
            u = m["unique"]
            print(f"-- {score} (n={m['n']:,})")
            print(
                f"   Spearman(score, |error|) = {m['spearman']:.3f}  "
                f"[{lo:.3f}, {hi:.3f}]  (random |rho| < {m['permutation_95']:.3f})"
            )
            print(
                f"   AUROC top-quartile error = {m['auroc']:.3f}  [{alo:.3f}, {ahi:.3f}]"
            )
            print(f"   sparsification gain      = {m['sparsification_gain']:.3f}")
            print(f"   mean |error| by quartile: {m['error_by_quartile']}")
            print(
                "   UNIQUE: AUC difference = "
                f"{u['auc_difference']:.3f} | performance drop all/low = "
                f"{u['performance_drop_all_vs_low']:.2f}, high/low = "
                f"{u['performance_drop_high_vs_low']:.2f} | increasing/decreasing "
                f"coefficient = {u['increasing_coefficient']:.2f}/"
                f"{u['decreasing_coefficient']:.2f}"
            )
            shifts = " · ".join(f"{k} {v:.2f}" for k, v in m["shifts"].items())
            print(f"   Spearman under shifts: {shifts}")


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
    parser.add_argument(
        "--black-box",
        default="all",
        choices=("all", *BLACK_BOXES),
        help="stand-in model whose errors are ranked (default: all)",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.csv).rename(
        columns={args.smiles_col: "smiles", args.y_col: "y"}
    )
    if len(df) > args.max_n:
        df = df.sample(n=args.max_n, random_state=SEED)
    boxes = BLACK_BOXES if args.black_box == "all" else (args.black_box,)
    report(
        Path(args.csv).stem,
        evaluate(df[["smiles", "y"]], name=args.y_col, black_boxes=boxes),
    )


if __name__ == "__main__":
    main()
