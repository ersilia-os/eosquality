"""Do the training scores rank a model's held-out errors? (Protocol B)

Reads the CSV written by ``scripts/evaluate_training_sets.py``. Left panel:
Spearman(score, |held-out error|) per endpoint, averaged over the three
stand-in models, with the two training scores side by side. Right panel: the
same split by black box, to separate the circular case (``rf_morgan``, the
surrogate's own family) from the transferring ones.

    python scripts/figures/training_validation.py
        [--results output/training_validation.csv]
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import stylia

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")

REPO = pathlib.Path(__file__).resolve().parents[2]
SCORES = ["trn_tanimoto_pct", "trn_difficulty"]
SCORE_LABELS = {"trn_tanimoto_pct": "Similarity", "trn_difficulty": "Difficulty"}
BOX_LABELS = {
    "rf_morgan": "RF Morgan\n(same family as surrogate)",
    "xgb_physchem": "XGBoost physchem",
    "knn_morgan": "5-NN Morgan",
}


def main():
    """Command-line entry point (see the module docstring for usage)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=pathlib.Path, default=REPO / "output/training_validation.csv"
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=REPO / "docs" / "figures"
    )
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    if not args.results.exists():
        raise SystemExit(
            f"{args.results} not found; run scripts/evaluate_training_sets.py first"
        )
    df = pd.read_csv(args.results)
    df = df[df["score"].isin(SCORES)]
    colors = stylia.CategoricalPalette("ersilia").get(len(SCORES))

    fig, axs = stylia.create_figure(1, 2, height=0.45)

    # Left: per endpoint, averaged over black boxes, continuous first.
    ax = axs.next()
    per_endpoint = df.pivot_table(
        index=["binary", "endpoint"], columns="score", values="spearman"
    ).reset_index()
    per_endpoint = per_endpoint.sort_values(["binary", "trn_difficulty"])
    y = np.arange(len(per_endpoint))
    height = 0.38
    for offset, (score, color) in enumerate(zip(SCORES, colors, strict=True)):
        ax.barh(
            y + (offset - 0.5) * height,
            per_endpoint[score],
            height=height,
            color=color,
            label=SCORE_LABELS[score],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{e} ({'binary' if b else 'continuous'})"
            for b, e in zip(
                per_endpoint["binary"], per_endpoint["endpoint"], strict=True
            )
        ],
        fontsize=5,
    )
    ax.axvline(0, color="black", lw=0.6)
    ax.legend(loc="lower right", fontsize=5, frameon=False)
    stylia.label(
        ax,
        xlabel="Spearman(score, |held-out error|)",
        ylabel="",
        title="Per endpoint (mean over black boxes)",
    )

    # Right: by black box and label kind.
    ax = axs.next()
    grouped = df.groupby(["black_box", "binary", "score"])["spearman"].mean()
    boxes = [b for b in BOX_LABELS if b in df["black_box"].unique()]
    labels, positions = [], []
    pos = 0
    for box in boxes:
        for binary in (False, True):
            for offset, (score, color) in enumerate(zip(SCORES, colors, strict=True)):
                value = grouped.get((box, binary, score), np.nan)
                ax.bar(pos + offset * 0.4, value, width=0.38, color=color)
            labels.append(f"{BOX_LABELS[box]}\n{'binary' if binary else 'continuous'}")
            positions.append(pos + 0.2)
            pos += 1.2
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=4.5)
    ax.axhline(0, color="black", lw=0.6)
    stylia.label(
        ax,
        xlabel="",
        ylabel="Spearman (mean over endpoints)",
        title="By stand-in model",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stylia.save_figure(str(args.out_dir / f"training_validation{args.suffix}.png"))


if __name__ == "__main__":
    main()
