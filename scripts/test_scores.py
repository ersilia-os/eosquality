#!/usr/bin/env python
"""Plot the four eosquality scores (typicality, extremity, support, consistency)
from one or more ``scores_*.csv`` files produced by ``eosquality run``.

Plots are organized by EOS identifier under ``outputs/<eos_id>/``. For each
input CSV named ``scores_<dataset>_<eos_id>_v<N>.csv``, two PNGs are written:

- ``outputs/<eos_id>/<dataset>_distributions.png`` — 2×2 score histograms.
- ``outputs/<eos_id>/<dataset>_pairwise.png`` — 4×4 corner plot.

CSVs whose names don't carry a recognizable EOS identifier are placed under
``outputs/unknown/`` instead of being skipped.

Usage:
    python scripts/plot_scores.py                            # all outputs/scores_*.csv
    python scripts/plot_scores.py outputs/scores_foo.csv     # one file
    python scripts/plot_scores.py outputs/scores_*.csv       # several files
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLS = ["typicality", "extremity", "support", "consistency"]

# Matches the EOS-style identifiers seen in the example data
# (eos4e40, eos7m30, eos3804, eos42ez …) — 4 alphanumeric chars after "eos".
_EOS_RE = re.compile(r"eos[a-z0-9]{4}", re.IGNORECASE)


def split_stem(stem: str) -> tuple[str, str]:
    """Split a scores-CSV stem into (eos_id, dataset_label).

    ``scores_molecules_1000_eos4e40_v1`` → ``("eos4e40", "molecules_1000")``
    ``scores_emh_paper_head_eos7m30_v1`` → ``("eos7m30", "emh_paper_head")``
    Anything without a match falls back to ``("unknown", stem)``.
    """
    m = _EOS_RE.search(stem)
    if not m:
        return "unknown", stem

    eos_id = m.group(0).lower()
    label = stem
    # Strip leading "scores_" if present.
    if label.startswith("scores_"):
        label = label[len("scores_") :]
    # Drop the trailing "_<eos_id>" (and any "_v<digits>" version suffix
    # that follows it).
    label = re.sub(
        rf"_{re.escape(eos_id)}(?:_v\d+)?$",
        "",
        label,
        flags=re.IGNORECASE,
    )
    if not label:
        label = "scores"
    return eos_id, label


# -----------------------------------------------------------------------------
# Palette
# -----------------------------------------------------------------------------


def get_palette() -> dict[str, str]:
    """One Ersilia brand color per score; falls back to hardcoded hex."""
    try:
        from stylia.colors.colors import NamedColors

        nc = NamedColors()

        def to_hex(rgb: tuple[float, float, float]) -> str:
            return "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )

        return {
            "typicality": to_hex(nc.get("blue")),
            "extremity": to_hex(nc.get("orange")),
            "support": to_hex(nc.get("green")),
            "consistency": to_hex(nc.get("purple")),
        }
    except Exception:
        return {
            "typicality": "#1d6996",
            "extremity": "#e17c05",
            "support": "#0f8554",
            "consistency": "#94346e",
        }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_distributions(
    df: pd.DataFrame,
    palette: dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    bins = np.linspace(0, 1, 51)
    for ax, col in zip(axes.flat, SCORE_COLS):
        v = df[col].dropna().to_numpy()
        ax.hist(
            v,
            bins=bins,
            density=True,
            color=palette[col],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
        )
        med = float(np.median(v))
        ax.axvline(med, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(
            f"{col}  (n={len(v):,}, μ={v.mean():.2f}±{v.std():.2f}, median={med:.2f})",
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylabel("Density")
    for ax in axes[-1, :]:
        ax.set_xlabel("score")
    fig.suptitle(title, y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_pairwise(
    df: pd.DataFrame,
    palette: dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    """4×4 corner plot. Diagonal: histograms. Lower: scatter. Upper: Pearson r."""
    import matplotlib.pyplot as plt

    n = len(SCORE_COLS)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), sharex="col")
    bins = np.linspace(0, 1, 41)

    for i, row_col in enumerate(SCORE_COLS):
        for j, col_col in enumerate(SCORE_COLS):
            ax = axes[i, j]
            if i == j:
                v = df[row_col].dropna().to_numpy()
                ax.hist(
                    v,
                    bins=bins,
                    density=True,
                    color=palette[row_col],
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.set_yticks([])
                ax.set_xlim(0, 1)
            elif i > j:
                x = df[col_col].to_numpy()
                y = df[row_col].to_numpy()
                ax.scatter(
                    x,
                    y,
                    s=6,
                    alpha=0.25,
                    color=palette[row_col],
                    edgecolor="none",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                x = df[col_col].to_numpy()
                y = df[row_col].to_numpy()
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 1 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                else:
                    r = float("nan")
                ax.text(
                    0.5,
                    0.5,
                    f"r = {r:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if j == 0 and i != j:
                ax.set_ylabel(row_col)
            else:
                ax.tick_params(left=False, labelleft=False)
            if i == n - 1:
                ax.set_xlabel(col_col)

    fig.suptitle(title, y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def process(csv_path: Path, palette: dict[str, str], output_root: Path) -> None:
    print(f"Reading {csv_path} ...", flush=True)
    df = pd.read_csv(csv_path)
    missing = [c for c in SCORE_COLS if c not in df.columns]
    if missing:
        print(
            f"  skipping {csv_path}: missing expected score columns {missing}",
            flush=True,
        )
        return
    print(f"  {len(df):,} rows, columns: {list(df.columns)}", flush=True)

    eos_id, dataset = split_stem(csv_path.stem)
    out_dir = output_root / eos_id
    out_dir.mkdir(parents=True, exist_ok=True)
    title_base = f"{eos_id} · {dataset}"
    print(f"  → {out_dir}/  (eos_id={eos_id}, dataset={dataset})", flush=True)

    plot_distributions(
        df,
        palette,
        out_dir / f"{dataset}_distributions.png",
        title=f"Score distributions — {title_base}",
    )
    plot_pairwise(
        df,
        palette,
        out_dir / f"{dataset}_pairwise.png",
        title=f"Pairwise scores — {title_base}",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "csvs",
        type=Path,
        nargs="*",
        help="Scores CSV(s). Defaults to outputs/scores_*.csv if omitted.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root folder for plot subfolders (default: outputs/).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    csvs = args.csvs
    if not csvs:
        csvs = [Path(p) for p in sorted(glob.glob("outputs/scores_*.csv"))]
    if not csvs:
        print(
            "error: no scores CSVs to plot (none passed, and outputs/scores_*.csv is empty).",
            file=sys.stderr,
        )
        return 1

    palette = get_palette()
    for csv_path in csvs:
        if not csv_path.is_file():
            print(f"  skipping {csv_path}: not a file", flush=True)
            continue
        process(csv_path, palette, args.output_root)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
