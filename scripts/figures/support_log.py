"""Support on two scales: calibrated support (0–1) vs support_log = −log10(support).

Support depends only on chemistry, so one model's CSVs are enough. Left:
calibrated support per query set — far-out sets collapse onto 0. Right:
the same values as support_log, where they separate; dashed lines mark
the 1%, 0.1% tails of same-size library molecules.

    python scripts/figures/support_log.py [--scores-dir output/]
"""

import numpy as np
import stylia
from _common import QUERY_SET_LABELS, QUERY_SETS, load_scores, parse_args, present

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")

N_BINS = 40
HALF_WIDTH = 0.42


def plot_violins(ax, values_by_set, colors, lo, hi):
    bins = np.linspace(lo, hi, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for pos, (values, color) in enumerate(zip(values_by_set, colors, strict=True), 1):
        h = np.histogram(values[np.isfinite(values)], bins=bins)[0]
        half = h / (h.max() or 1) * HALF_WIDTH
        ax.fill_between(centers, pos - half, pos + half, step="mid", color=color)
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(values_by_set) + 0.5, 0.5)
    ax.set_yticks(range(1, len(values_by_set) + 1))


def main():
    args = parse_args(__doc__)
    df = load_scores(args.scores_dir)
    if "support_log" not in df.columns:
        raise SystemExit(
            "These score CSVs have no support_log column (artifact format < 3)."
        )
    df = df[df["model"] == sorted(df["model"].unique())[0]]
    sets = present(QUERY_SETS, df["query_set"].unique())
    colors = stylia.CategoricalPalette("ersilia").get(len(sets))
    labels = [QUERY_SET_LABELS.get(s, s) for s in sets]

    fig, axs = stylia.create_figure(1, 2)
    ax = axs.next()
    plot_violins(
        ax,
        [df.loc[df.query_set == s, "support"].to_numpy(float) for s in sets],
        colors,
        0,
        1,
    )
    ax.set_yticklabels(labels)
    stylia.label(
        ax, xlabel="Support (calibrated)", ylabel="", title="Linear scale", abc="A"
    )

    ax = axs.next()
    values = [df.loc[df.query_set == s, "support_log"].to_numpy(float) for s in sets]
    hi = float(np.ceil(np.nanmax(np.concatenate(values))))
    plot_violins(ax, values, colors, 0, hi)
    for x in (2, 3):
        ax.axvline(x, color=stylia.ErsiliaColors().gray, linestyle="--")
    ax.set_yticklabels([])
    stylia.label(
        ax,
        xlabel="support_log = −log10(support)",
        ylabel="",
        title="Log scale",
        abc="B",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stylia.save_figure(str(args.out_dir / f"support_log{args.suffix}.png"))


if __name__ == "__main__":
    main()
