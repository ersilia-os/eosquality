"""Training-score distributions per query set (rows) and model (violins).

Left column: ``trn_tanimoto_pct`` (calibrated, 0–1). Right: ``trn_tanimoto_raw``
(mean Tanimoto to the 5 nearest training molecules). Only models fitted with
training sets appear.

A query set drawn from the model's own training chemistry should sit near
0.5 on the calibrated scales; sets of unrelated chemistry shift towards 1.

    python scripts/figures/training_scores.py [--scores-dir output/]
"""

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
from score_distributions import plot_violins

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")

PANELS = [
    ("trn_tanimoto_pct", "Similarity (percentile)", (0.0, 1.0)),
    ("trn_tanimoto_raw", "Similarity (raw, mean Tanimoto)", (0.0, 1.0)),
]


def main():
    """Command-line entry point (see the module docstring for usage)."""
    args = parse_args(__doc__)
    df = load_scores(args.scores_dir)
    panels = [p for p in PANELS if p[0] in df.columns]
    if not panels:
        raise SystemExit(f"No trn_* columns in the score CSVs of {args.scores_dir}")
    with_training = [
        m
        for m in present(MODELS, df["model"].unique())
        if df.loc[df["model"] == m, panels[0][0]].notna().any()
    ]
    rows = present(QUERY_SETS, df["query_set"].unique())
    colors = stylia.CategoricalPalette("ersilia").get(len(with_training))

    fig, axs = stylia.create_figure(len(rows), len(panels), height=0.75)
    for i, qs in enumerate(rows):
        sub = df[df["query_set"] == qs]
        for j, (column, title, (lo, hi)) in enumerate(panels):
            ax = axs.next()
            values = [
                sub.loc[sub["model"] == m, column].to_numpy(float)
                for m in with_training
            ]
            plot_violins(ax, values, colors, lo, hi)
            if column != "trn_tanimoto_raw":
                ax.axvline(0.5, color="black", lw=0.6, ls="--", alpha=0.5)
            ax.set_yticklabels(
                [MODEL_LABELS.get(m, m) for m in with_training] if j == 0 else []
            )
            stylia.label(
                ax,
                xlabel=column if i == len(rows) - 1 else "",
                ylabel=QUERY_SET_LABELS.get(qs, qs) if j == 0 else "",
                title=title if i == 0 else "",
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stylia.save_figure(str(args.out_dir / f"training_scores{args.suffix}.png"))


if __name__ == "__main__":
    main()
