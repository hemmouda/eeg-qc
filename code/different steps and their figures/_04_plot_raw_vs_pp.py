from pathlib import Path

import matplotlib.pyplot as plt

from _03_raw_vs_pp import OUTPUT
from core.consts import TARGETS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("_04_raw_vs_pp_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = False
SAVE_DPI = 300

FONT_SIZE = 12.5

TEXTWIDTH_INCH = 5.455853 * 1.5
FIG_HEIGHT = 0.55 * TEXTWIDTH_INCH

COLORS = {
    "raw": "#1f77b4",
    "pp": "#ff7f0e",
}

# -----------------------------------------------------------------------------
# X-axis label mapping (robust display layer)
# -----------------------------------------------------------------------------

TARGET_LABELS = {
    "artf": "artf_ss",
    "musc": "musc_ss",
    "eyem": "eyem_ss",
    "elec": "elec_ss",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_macro_f1(results: dict, target: str) -> tuple[float, float]:
    raw = results[target]["raw"]["macro avg"]["f1-score"]
    pp = results[target]["pp"]["macro avg"]["f1-score"]
    return raw, pp


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
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


def plot():
    targets = list(TARGETS)

    raw_scores = []
    pp_scores = []

    for t in targets:
        raw_f1, pp_f1 = extract_macro_f1(OUTPUT, t)
        raw_scores.append(raw_f1)
        pp_scores.append(pp_f1)

    x = range(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_INCH, FIG_HEIGHT))

    bars_raw = ax.bar(
        [i - width / 2 for i in x],
        raw_scores,
        width=width,
        color=COLORS["raw"],
        label="Raw data",
    )

    bars_pp = ax.bar(
        [i + width / 2 for i in x],
        pp_scores,
        width=width,
        color=COLORS["pp"],
        label="Filtered data",
    )

    annotate_bars(ax, bars_raw)
    annotate_bars(ax, bars_pp)

    # X-axis (robust mapping)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [TARGET_LABELS.get(t, t) for t in targets],
        fontsize=FONT_SIZE,
    )

    ax.set_ylim(top=0.8)
    ax.set_xlabel("Target", fontsize=FONT_SIZE)
    ax.set_ylabel("Macro F1-score", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    ax.legend(
        fontsize=FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 1])

    output_path = OUTPUT_DIR / "raw_vs_pp_macro_f1_bar.png"
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")

    print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot()
