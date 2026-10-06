"""Score distributions per query set (rows), score (columns) and model (violins).

Each panel stacks one horizontal histogram-violin per model; the x-axis is
the calibrated score in [0, 1] (or the raw value with ``--raw``, on a range
shared down each column). Library-sample molecules should look flat
(uniform); query sets far from the reference shift towards 0 for support,
and wherever the model behaves unlike it does on the reference for the
output-based scores.

    python scripts/figures/score_distributions.py [--scores-dir output/] [--raw]
"""

import sys

import numpy as np
import stylia
from _common import (
    MODEL_LABELS,
    MODELS,
    QUERY_SET_LABELS,
    QUERY_SETS,
    SCORES,
    load_scores,
    parse_args,
    present,
)

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")

N_BINS = 30
HALF_WIDTH = 0.42


def plot_violins(ax, values_by_model, colors, lo, hi):
    """Stepped, peak-normalised histogram silhouettes, first model on top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    values_by_model : list of numpy.ndarray
        Values per row, top to bottom.
    colors : list
        One colour per row.
    lo, hi : float
        x-axis range.
    """
    bins = np.linspace(lo, hi, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for pos, (values, color) in enumerate(zip(values_by_model, colors, strict=True), 1):
        v = values[np.isfinite(values)]
        h = np.histogram(v, bins=bins)[0] if v.size else np.zeros(N_BINS)
        half = h / (h.max() or 1) * HALF_WIDTH
        ax.fill_between(centers, pos - half, pos + half, step="mid", color=color)
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(values_by_model) + 0.5, 0.5)
    ax.set_yticks(range(1, len(values_by_model) + 1))


def main():
    """Command-line entry point (see the module docstring for usage)."""
    raw = "--raw" in sys.argv
    if raw:
        sys.argv.remove("--raw")
    args = parse_args(__doc__)
    df = load_scores(args.scores_dir)
    rows = present(QUERY_SETS, df["query_set"].unique())
    models = present(MODELS, df["model"].unique())
    scores = [s for s in SCORES if s in df.columns]
    colors = stylia.CategoricalPalette("ersilia").get(len(models))

    # A 5 × 5 grid of stacked violins is unreadable at the default slide
    # height (0.3 × width), so this figure is taller than stylia's default.
    fig, axs = stylia.create_figure(len(rows), len(scores), height=0.75)
    for i, qs in enumerate(rows):
        sub = df[df["query_set"] == qs]
        for j, score in enumerate(scores):
            ax = axs.next()
            col = f"{score}_raw" if raw else score
            if raw:
                finite = df[col].to_numpy(float)
                finite = finite[np.isfinite(finite)]
                lo, hi = float(finite.min()), float(finite.max()) or 1.0
            else:
                lo, hi = 0.0, 1.0
            values = [sub.loc[sub["model"] == m, col].to_numpy(float) for m in models]
            plot_violins(ax, values, colors, lo, hi)
            ax.set_yticklabels(
                [MODEL_LABELS.get(m, m) for m in models] if j == 0 else []
            )
            stylia.label(
                ax,
                xlabel=("raw" if raw else "calibrated") if i == len(rows) - 1 else "",
                ylabel=QUERY_SET_LABELS.get(qs, qs) if j == 0 else "",
                title=score.capitalize() if i == 0 else "",
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "score_distributions_raw" if raw else "score_distributions"
    stylia.save_figure(str(args.out_dir / f"{name}{args.suffix}.png"))


if __name__ == "__main__":
    main()
