"""Score distributions per query set (rows) and model (violins), four figures.

One figure each for the reference and the training scores, in their
percentile and raw form: ``reference_pct``, ``reference_raw``,
``training_pct`` and ``training_raw`` (PNG in ``docs/figures/``). Every panel
stacks one histogram-violin per model; the columns are the scores of the
figure. Library-sample molecules should look flat on the reference
percentiles; query sets shift wherever the model behaves unlike it does on
the reference. Higher training scores mean closer to the training sets. Only
models fitted with training sets appear in the training figures. The x-range
of a score is shared down its column.

    python scripts/figures/score_distributions.py [--scores-dir output/]
"""

import numpy as np
import stylia
from _common import (
    MODEL_LABELS,
    MODELS,
    QUERY_SET_LABELS,
    QUERY_SETS,
    load_scores,
    parse_args,
    present,
)

stylia.set_format("slide")
stylia.set_style("ersilia")

N_BINS = 30
HALF_WIDTH = 0.42

# figure name -> (score columns in the CSVs, panel titles)
FIGURES = {
    "reference_pct": (
        ["ref_typicality_pct", "ref_extremity_pct"],
        ["Typicality (percentile)", "Extremity (percentile)"],
    ),
    "reference_raw": (
        ["ref_typicality_raw", "ref_extremity_raw"],
        ["Typicality (raw)", "Extremity (raw)"],
    ),
    "training_pct": (
        ["trn_tanimoto_pct", "trn_physchem_pct"],
        ["Tanimoto similarity (percentile)", "Physchem similarity (percentile)"],
    ),
    "training_raw": (
        ["trn_tanimoto_raw", "trn_physchem_raw"],
        ["Tanimoto similarity (raw)", "Physchem similarity (raw)"],
    ),
}


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


def _x_range(df, column):
    """The x-range of a score: [0, 1] for percentiles, the data range for raw."""
    if column.endswith("_pct"):
        return 0.0, 1.0
    finite = df[column].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    return float(finite.min()), float(finite.max())


def draw(df, columns, titles, path):
    """One figure: query sets down, scores across, a violin per model.

    Parameters
    ----------
    df : pandas.DataFrame
        The long score table of :func:`_common.load_scores`.
    columns : list of str
        Score columns, one per panel column.
    titles : list of str
        Panel titles.
    path : pathlib.Path
        PNG to write.
    """
    models = [
        m
        for m in present(MODELS, df["model"].unique())
        if df.loc[df["model"] == m, columns[0]].notna().any()
    ]
    rows = present(QUERY_SETS, df["query_set"].unique())
    colors = stylia.CategoricalPalette("npg").get(len(models))
    # A 5 × 2 grid of stacked violins is unreadable at the default slide
    # height (0.3 × width), so these figures are taller than stylia's default.
    fig, axs = stylia.create_figure(len(rows), len(columns), height=0.75)
    ranges = [_x_range(df, c) for c in columns]
    for i, qs in enumerate(rows):
        sub = df[df["query_set"] == qs]
        for j, column in enumerate(columns):
            ax = axs.next()
            values = [
                sub.loc[sub["model"] == m, column].to_numpy(float) for m in models
            ]
            plot_violins(ax, values, colors, *ranges[j])
            ax.set_yticklabels(
                [MODEL_LABELS.get(m, m) for m in models] if j == 0 else []
            )
            stylia.label(
                ax,
                xlabel=column if i == len(rows) - 1 else "",
                ylabel=QUERY_SET_LABELS.get(qs, qs) if j == 0 else "",
                title=titles[j] if i == 0 else "",
            )
    stylia.save_figure(str(path))


def main():
    """Command-line entry point (see the module docstring for usage)."""
    args = parse_args(__doc__)
    df = load_scores(args.scores_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (columns, titles) in FIGURES.items():
        if not all(c in df.columns for c in columns):
            print(f"skipping {name}: {columns} not in the score CSVs")
            continue
        draw(df, columns, titles, args.out_dir / f"{name}{args.suffix}.png")


if __name__ == "__main__":
    main()
