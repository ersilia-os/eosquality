"""Run the training-score method check over the example training sets.

Wraps ``evaluate_training.py`` (Protocol B: scaffold-split one training set,
fit the training modality on the 80%, ask whether the training scores rank a
stand-in model's errors on the held-out 20%) and runs it over every
``data/training_examples/training_<eos>_v1/<column>.csv``, writing one tidy
CSV of metrics.

The result is the evidence behind the training-mode numbers in
``docs/status.md``; ``scripts/figures/training_validation.py`` plots it.

    python scripts/evaluate_training_sets.py [--out output/training_validation.csv]
        [--endpoints eos7m30:herg,eos4e40:inhibition_50um] [--max-n 10000]

Resumable: endpoints already in ``--out`` are skipped, so an interrupted run
continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_training as ev  # noqa: E402

TRAINING_DIR = REPO / "data" / "training_examples"
DEFAULT_OUT = REPO / "output" / "training_validation.csv"
# Endpoints of the example models, chosen for a spread of sizes, label kinds
# and assay types. "all" evaluates every training file of every example model.
DEFAULT_ENDPOINTS = [
    "eos4e40:inhibition_50um",
    "eos42ez:cytotoxicity_imr90",
    "eos7m30:ppbr_az",
    "eos7m30:clearance_hepatocyte_az",
    "eos7m30:hydrationfreeenergy_freesolv",
    "eos7m30:vdss_lombardo",
    "eos3804:abaumannii_inhibition_probability",
    "eos42ez:cytotoxicity_hepg2",
    "eos7m30:herg",
    "eos7m30:ames",
    "eos7m30:bbb_martins",
    "eos7m30:cyp3a4_veith",
    "eos7m30:dili",
    "eos7m30:solubility_aqsoldb",
    "eos7m30:lipophilicity_astrazeneca",
    "eos7m30:caco2_wang",
    "eos7m30:ld50_zhu",
    "eos7m30:half_life_obach",
    "eos7m30:clearance_microsome_az",
]


def all_endpoints() -> list[str]:
    """Every ``<eos>:<column>`` training file under ``data/training_examples/``.

    Returns
    -------
    list of str
    """
    out = []
    for folder in sorted(TRAINING_DIR.glob("training_*_v1")):
        eos = folder.name.split("_")[-2]
        out += [f"{eos}:{csv.stem}" for csv in sorted(folder.glob("*.csv"))]
    return out


def evaluate_one(eos: str, column: str, max_n: int) -> list[dict]:
    """Metrics rows for one endpoint (one per black box and training score).

    Parameters
    ----------
    eos : str
        Model id, e.g. ``"eos7m30"``.
    column : str
        Output column (the training file's stem).
    max_n : int
        Random subsample of the training set before splitting.

    Returns
    -------
    list of dict
    """
    path = TRAINING_DIR / f"training_{eos}_v1" / f"{column}.csv"
    df = pd.read_csv(path)
    label = next((c for c in ("y", "value") if c in df.columns), None)
    if label is None:
        raise SystemExit(f"{path} has no 'y' or 'value' column")
    df = df.rename(columns={label: "y"})
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=ev.SEED)
    result = ev.evaluate(df[["smiles", "y"]], name=column)
    rows = []
    for box, per_score in result["black_boxes"].items():
        for score, m in per_score.items():
            rows.append(
                {
                    "model": eos,
                    "endpoint": column,
                    "binary": result["binary"],
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "black_box": box,
                    "score": score,
                    "spearman": m["spearman"],
                    "spearman_lo": m["spearman_ci"][0],
                    "spearman_hi": m["spearman_ci"][1],
                    "permutation_95": m["permutation_95"],
                    "auroc": m["auroc"],
                    "sparsification_gain": m["sparsification_gain"],
                    "error_by_quartile": json.dumps(m["error_by_quartile"]),
                }
            )
    return rows


def main() -> None:
    """Command-line entry point (see the module docstring for usage)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--endpoints",
        default=",".join(DEFAULT_ENDPOINTS),
        help="comma-separated <eos>:<column>, or 'all'",
    )
    parser.add_argument("--max-n", type=int, default=10_000)
    args = parser.parse_args()

    endpoints = (
        all_endpoints() if args.endpoints == "all" else args.endpoints.split(",")
    )
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.out.exists():
        rows = pd.read_csv(args.out).to_dict("records")
        done = {(r["model"], r["endpoint"]) for r in rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for item in endpoints:
        eos, column = item.split(":")
        if (eos, column) in done:
            print(f"{item}: already in {args.out}, skipped", flush=True)
            continue
        t0 = time.time()
        rows += evaluate_one(eos, column, args.max_n)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"{item}: {time.time() - t0:.0f}s", flush=True)
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
