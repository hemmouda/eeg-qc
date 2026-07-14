from pathlib import Path

import matplotlib.pyplot as plt

from _05_full_hgb_perf import OUTPUT as HGB_OUTPUT
from _12_full_dt_perf import OUTPUT as DT_OUTPUT

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("_13_full_dt_perf_vs_full_hgb_pref_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = True
SAVE_DPI = 300

FONT_SIZE = 12

TEXTWIDTH_INCH = 5.455853 * 1.375
FIG_HEIGHT = 0.64 * TEXTWIDTH_INCH

COLORS = {
    "class0": "#1f77b4",
    "macro": "#2ca02c",
    "class1": "#ff7f0e",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 2,
        )


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------


def plot_target(target: str):

    models = ["Full HGB", "Full DT (depth=6)"]

    class0_scores = [
        HGB_OUTPUT[target]["0"]["f1-score"],
        DT_OUTPUT[target]["0"]["f1-score"],
    ]

    macro_scores = [
        HGB_OUTPUT[target]["macro avg"]["f1-score"],
        DT_OUTPUT[target]["macro avg"]["f1-score"],
    ]

    class1_scores = [
        HGB_OUTPUT[target]["1"]["f1-score"],
        DT_OUTPUT[target]["1"]["f1-score"],
    ]

    x = range(len(models))
    width = 0.25
    width = 0.1375

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_INCH, FIG_HEIGHT))

    bars_class0 = ax.bar(
        [i - width for i in x],
        class0_scores,
        width=width,
        color=COLORS["class0"],
        label="Class 0 F1-score",
    )

    bars_macro = ax.bar(
        x,
        macro_scores,
        width=width,
        color=COLORS["macro"],
        label="Macro F1-score",
    )

    bars_class1 = ax.bar(
        [i + width for i in x],
        class1_scores,
        width=width,
        color=COLORS["class1"],
        label="Class 1 F1-score",
    )

    annotate_bars(ax, bars_class0)
    annotate_bars(ax, bars_macro)
    annotate_bars(ax, bars_class1)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=FONT_SIZE)

    ax.set_xlabel("Model", fontsize=FONT_SIZE)
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

    output_path = OUTPUT_DIR / f"{target}_full_hgb_vs_full_dt.png"

    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")

    print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    for target in HGB_OUTPUT:
        plot_target(target)


if __name__ == "__main__":
    main()
