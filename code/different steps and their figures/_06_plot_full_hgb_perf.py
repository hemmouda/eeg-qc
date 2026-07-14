from pathlib import Path

import matplotlib.pyplot as plt

from _05_full_hgb_perf import OUTPUT
from core.consts import TARGETS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("_06_plot_full_hgb_perf_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = True
SAVE_DPI = 300

FONT_SIZE = 12
FONT_SIZE = 13

TEXTWIDTH_INCH = 5.455853 * 1.5
FIG_HEIGHT = 0.65 * TEXTWIDTH_INCH

COLORS = {
    "class0": "#1f77b4",
    "macro": "#2ca02c",
    "class1": "#ff7f0e",
}

TARGET_LABELS = {
    "artf": "artf_ss",
    "musc": "musc_ss",
    "eyem": "eyem_ss",
    "elec": "elec_ss",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_scores(results: dict, target: str):
    class0 = results[target]["0"]["f1-score"]
    class1 = results[target]["1"]["f1-score"]
    macro = results[target]["macro avg"]["f1-score"]
    return class0, macro, class1


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 2,
        )


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------


def plot():
    targets = list(TARGETS)

    class0_scores = []
    macro_scores = []
    class1_scores = []

    for t in targets:
        c0, macro, c1 = extract_scores(OUTPUT, t)
        class0_scores.append(c0)
        macro_scores.append(macro)
        class1_scores.append(c1)

    x = range(len(targets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_INCH, FIG_HEIGHT))

    bars_c0 = ax.bar(
        [i - width for i in x],
        class0_scores,
        width=width,
        color=COLORS["class0"],
        label="Class 0 F1-score",
    )

    bars_macro = ax.bar(
        list(x),
        macro_scores,
        width=width,
        color=COLORS["macro"],
        label="Macro F1-score",
    )

    bars_c1 = ax.bar(
        [i + width for i in x],
        class1_scores,
        width=width,
        color=COLORS["class1"],
        label="Class 1 F1-score",
    )

    annotate_bars(ax, bars_c0)
    annotate_bars(ax, bars_macro)
    annotate_bars(ax, bars_c1)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [TARGET_LABELS.get(t, t) for t in targets],
        fontsize=FONT_SIZE,
    )

    ax.set_xlabel("Target", fontsize=FONT_SIZE)
    ax.set_ylabel("F1-score", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    ax.legend(
        fontsize=FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 1])

    output_path = OUTPUT_DIR / "full_hgb_perf_f1_scores.png"
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")

    print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot()
