"""Redundancy check: Spearman correlation between the calibrated scores.

One heatmap per model, computed over all of that model's queries (every
query set pooled). Off-diagonal values near 0 mean the scores carry
distinct information; values near ±1 flag redundant scores.

    python scripts/figures/score_correlations.py [--scores-dir output/]
"""

import numpy as np
import stylia
from _common import MODEL_LABELS, MODELS, SCORES, load_scores, parse_args, present

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")

ABBREV = {
    "typicality": "Typ",
    "extremity": "Ext",
}


def plot_heatmap(ax, corr, labels, cmap):
    """Draw an annotated correlation heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    corr : numpy.ndarray
        Square correlation matrix.
    labels : list of str
        Row/column labels.
    cmap : stylia.DivergingColormap
        Fitted colormap.
    """
    colors = np.asarray(cmap.transform(corr.ravel())).reshape(*corr.shape, -1)
    ax.imshow(colors)
    for (r, c), v in np.ndenumerate(corr):
        luminance = colors[r, c][:3] @ [0.299, 0.587, 0.114]
        ax.text(
            c,
            r,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if luminance < 0.5 else "black",
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)


def main():
    """Command-line entry point (see the module docstring for usage)."""
    args = parse_args(__doc__)
    df = load_scores(args.scores_dir)
    models = present(MODELS, df["model"].unique())
    scores = [s for s in SCORES if s in df.columns]
    labels = [ABBREV.get(s, s) for s in scores]
    cmap = stylia.DivergingColormap("plum_mint")
    cmap.fit([-1.0, 1.0])

    fig, axs = stylia.create_figure(1, len(models))
    for i, model in enumerate(models):
        ax = axs.next()
        corr = df.loc[df["model"] == model, scores].corr(method="spearman").to_numpy()
        plot_heatmap(ax, corr, labels, cmap)
        stylia.label(
            ax,
            xlabel="",
            ylabel="",
            title=f"{model} ({MODEL_LABELS.get(model, '')})",
            abc="ABCDE"[i],
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stylia.save_figure(str(args.out_dir / f"score_correlations{args.suffix}.png"))


if __name__ == "__main__":
    main()
