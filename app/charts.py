"""Benchmark charts (matplotlib, transparent background for light and dark themes)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.data import RESULTS
from app.settings import AXIS_COLOR, SERIES_COLORS

METRIC_LABELS = {"f1": "F1", "exact_match": "Exact Match"}
BENCHMARK_PRECISIONS = ["FP32", "BF16", "INT8", "INT4"]


def _style_axes(ax):
    ax.set_facecolor("none")
    ax.tick_params(colors=AXIS_COLOR)
    ax.yaxis.label.set_color(AXIS_COLOR)
    ax.title.set_color(AXIS_COLOR)
    ax.grid(axis="y", color=AXIS_COLOR, alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS_COLOR)


def results_figure(metric: str = "f1"):
    """Scores by precision (bars) and change vs FP32 (lines), one series per model."""
    label = METRIC_LABELS[metric]
    models = RESULTS["model"].unique()
    width = 0.8 / len(models)

    fig, (ax_score, ax_delta) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_alpha(0)
    for i, (model, color) in enumerate(zip(models, SERIES_COLORS)):
        scores = RESULTS[RESULTS["model"] == model].set_index("precision").loc[BENCHMARK_PRECISIONS, metric]
        xs = [p + (i - (len(models) - 1) / 2) * width for p in range(len(BENCHMARK_PRECISIONS))]
        bars = ax_score.bar(xs, scores, width * 0.92, label=model, color=color)
        ax_score.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, color=AXIS_COLOR)
        ax_delta.plot(BENCHMARK_PRECISIONS, scores - scores["FP32"], marker="o", color=color, linewidth=2)

    ax_score.set_xticks(range(len(BENCHMARK_PRECISIONS)), BENCHMARK_PRECISIONS)
    ax_score.set_ylim(0.3, 0.8)
    ax_score.set_ylabel(label)
    ax_score.set_title(f"{label} by precision", fontsize=11)
    ax_score.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=AXIS_COLOR, ncols=3)

    ax_delta.axhline(0, color=AXIS_COLOR, linewidth=1, linestyle="--")
    ax_delta.set_ylabel(f"{label} change vs FP32")
    ax_delta.set_title("Accuracy lost by quantizing", fontsize=11)

    for ax in (ax_score, ax_delta):
        _style_axes(ax)
    fig.tight_layout()
    return fig
