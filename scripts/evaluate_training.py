"""Method check for training-modality scores (Protocol B).

Given one training set (``smiles``, ``y``), split it 80/20 by Murcko
scaffold, fit eosquality's training modality on the 80% and a surrogate
model (random forest on Morgan bits) on the same 80%, then ask whether the
training scores of the held-out 20% predict the surrogate's error there.
Without an external labelled set this validates the *method*, not the
Ersilia model itself.

Reported (with 95% bootstrap intervals):
- Spearman(training_distance, |error|): > 0 means farther ↔ larger error.
- AUROC of training_distance for flagging the top-quartile errors.
- Mean |error| per distance quartile.

    python scripts/evaluate_training.py --csv train.csv [--y-col y] [--max-n 20000]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

from eosquality import ErsiliaQuality

RDLogger.DisableLog("rdApp.*")
SEED = 0


def scaffold_split(smiles: list[str], test_frac: float = 0.2) -> np.ndarray:
    """Boolean test mask: whole scaffold groups, random order, until ~test_frac."""
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


def morgan(smiles: list[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array(
        [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
        dtype=np.uint8,
    )


def bootstrap(fn, *arrays, n=500):
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(arrays[0]), len(arrays[0]))
        try:
            stats.append(fn(*(a[i] for a in arrays)))
        except ValueError:
            continue
    return np.percentile(stats, [2.5, 97.5])


def evaluate(df: pd.DataFrame, name: str = "column") -> dict:
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
        distance = eq.run(q).scores["training_distance"].to_numpy()

    X_tr, X_te = morgan(train_df.smiles.tolist()), morgan(test_df.smiles.tolist())
    if binary:
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
        model.fit(X_tr, train_df.y)
        err = np.abs(model.predict_proba(X_te)[:, 1] - test_df.y.to_numpy())
    else:
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
        model.fit(X_tr, train_df.y)
        err = np.abs(model.predict(X_te) - test_df.y.to_numpy())

    ok = np.isfinite(distance)
    distance, err = distance[ok], err[ok]
    rho = spearmanr(distance, err)[0]
    rho_ci = bootstrap(lambda d, e: spearmanr(d, e)[0], distance, err)
    big = err >= np.quantile(err, 0.75)
    auc = roc_auc_score(big, distance)
    auc_ci = bootstrap(lambda b, d: roc_auc_score(b, d), big, distance)
    quart = pd.qcut(
        distance, 4, labels=["Q1 (near)", "Q2", "Q3", "Q4 (far)"], duplicates="drop"
    )
    by_q = pd.Series(err).groupby(quart, observed=True).mean()
    return {
        "n_train": len(train_df),
        "n_test": int(ok.sum()),
        "binary": binary,
        "spearman": rho,
        "spearman_ci": rho_ci,
        "auroc": auc,
        "auroc_ci": auc_ci,
        "error_by_distance_quartile": by_q.round(4).to_dict(),
    }


def report(name: str, r: dict) -> None:
    print(
        f"\n== {name} | n_train={r['n_train']:,} n_test={r['n_test']:,} | "
        f"{'binary' if r['binary'] else 'continuous'} y"
    )
    print(
        f"Spearman(distance, |error|) = {r['spearman']:.3f}  95% CI [{r['spearman_ci'][0]:.3f}, {r['spearman_ci'][1]:.3f}]"
    )
    print(
        f"AUROC top-quartile error  = {r['auroc']:.3f}  95% CI [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}]"
    )
    print("mean |error| by distance quartile:", r["error_by_distance_quartile"])


def main() -> None:
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
