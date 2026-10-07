"""Calibration check: calibrated scores of library molecules vs Uniform(0, 1).

Each score is the reference CDF of a raw value, so molecules drawn from the
reference library should score ~Uniform(0, 1). One panel per score shows,
per model, the deviation of the empirical CDF from uniform (ECDF(x) − x);
the shaded band is the 95% Kolmogorov–Smirnov acceptance region for the
sample size, and the panel title gives the worst KS distance across models.
Run before and after a calibration change (``--suffix``) to compare.

    python scripts/figures/reference_calibration.py [--scores-dir output/]
"""

import numpy as np
import stylia
from _common import MODELS, SCORES, load_scores, model_label, parse_args, present
from scipy.stats import kstest

# Format: slide | Style: ersilia — change with stylia.set_format() / stylia.set_style()
stylia.set_format("slide")
stylia.set_style("ersilia")


def plot_ecdf_deviation(ax, values_by_model, colors):
    """Draw ECDF(x) − x per model; return the worst KS statistic.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    values_by_model : dict of str to numpy.ndarray
        Calibrated scores per model.
    colors : list
        One colour per model.

    Returns
    -------
    float
        Worst KS statistic across models.
    """
    n_min = min(int(np.isfinite(v).sum()) for v in values_by_model.values())
    band = 1.358 / np.sqrt(max(n_min, 1))  # 95% two-sided KS critical value
    ax.axhspan(-band, band, color=stylia.ErsiliaColors().gray, alpha=0.2)
    ax.axhline(0, color=stylia.ErsiliaColors().gray)
    worst = 0.0
    for (model, values), color in zip(values_by_model.items(), colors, strict=True):
        v = np.sort(values[np.isfinite(values)])
        if v.size == 0:
            continue
        worst = max(worst, kstest(v, "uniform").statistic)
        ax.step(
            v,
            np.arange(1, v.size + 1) / v.size - v,
            where="post",
            color=color,
            label=model_label(model),
        )
    ax.set_xlim(0, 1)
    return worst


def main():
    """Command-line entry point (see the module docstring for usage)."""
    args = parse_args(__doc__)
    df = load_scores(args.scores_dir)
    df = df[df["query_set"] == "molecules"]
    models = present(MODELS, df["model"].unique())
    scores = [s for s in SCORES if s in df.columns]
    colors = stylia.CategoricalPalette("ersilia").get(len(models))

    fig, axs = stylia.create_figure(1, len(scores))
    lim = 0.0
    panels = []
    for i, score in enumerate(scores):
        ax = axs.next()
        values = {m: df.loc[df["model"] == m, score].to_numpy(float) for m in models}
        worst = plot_ecdf_deviation(ax, values, colors)
        lim = max(lim, *(abs(line.get_ydata()).max() for line in ax.lines[1:]))
        stylia.label(
            ax,
            xlabel="Calibrated score",
            ylabel="ECDF − uniform" if i == 0 else "",
            title=f"{score.capitalize()} (KS ≤ {worst:.2f})",
            abc="ABCDE"[i],
        )
        panels.append(ax)
    for ax in panels:
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
    panels[0].legend(loc="lower left")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stylia.save_figure(str(args.out_dir / f"reference_calibration{args.suffix}.png"))


if __name__ == "__main__":
    main()
