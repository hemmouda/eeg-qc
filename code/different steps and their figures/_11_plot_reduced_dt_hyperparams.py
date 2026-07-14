from pathlib import Path
import matplotlib.pyplot as plt

from _10_reduced_dt_hyperparams import OUTPUT
from core.consts import TARGETS

# Copy pasted from 08


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("_11_reduced_dt_hyperparams_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = False
SAVE_DPI = 300

FONT_SIZE = 12
FONT_SIZE = 13

TEXTWIDTH_INCH = 5.455853 * 1.6
FIG_HEIGHT = 0.5 * TEXTWIDTH_INCH

COLORS = {
    "class0": "#1f77b4",
    "macro": "#2ca02c",
    "class1": "#ff7f0e",
}

DEPTHS = sorted(next(iter(OUTPUT.values())).keys())

TARGET_LABELS = {
    "artf": "artf_ss",
    "musc": "musc_ss",
    "eyem": "eyem_ss",
    "elec": "elec_ss",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_series(target_results: dict, metric: str) -> list[float]:
    return [target_results[d][metric]["f1-score"] for d in DEPTHS]


def annotate_line(ax, x, y, color):
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=FONT_SIZE - 2.5,
            color=color,
        )


# -----------------------------------------------------------------------------
# Plot per target
# -----------------------------------------------------------------------------


def plot_target(target: str, results: dict):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_INCH, FIG_HEIGHT))

    class0 = extract_series(results, "0")
    class1 = extract_series(results, "1")
    macro = extract_series(results, "macro avg")

    ax.plot(
        DEPTHS,
        class0,
        marker="o",
        color=COLORS["class0"],
        label="Class 0 F1-score",
    )

    ax.plot(
        DEPTHS,
        macro,
        marker="o",
        linestyle="--",
        color=COLORS["macro"],
        label="Macro F1-score",
    )

    ax.plot(
        DEPTHS,
        class1,
        marker="o",
        color=COLORS["class1"],
        label="Class 1 F1-score",
    )

    annotate_line(ax, DEPTHS, class0, COLORS["class0"])
    annotate_line(ax, DEPTHS, class1, COLORS["class1"])
    annotate_line(ax, DEPTHS, macro, COLORS["macro"])

    ax.set_xlabel("Tree Depth", fontsize=FONT_SIZE)
    ax.set_ylabel("F1-score", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    ax.set_xticks(DEPTHS)

    # grid (depth axis)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # legend on top
    ax.legend(
        fontsize=FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 1])

    out = OUTPUT_DIR / f"{target}_reduced_dt_depth_f1.png"
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")

    print(f"Saved: {out}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    for target in TARGETS:
        plot_target(target, OUTPUT[target])


if __name__ == "__main__":
    main()
