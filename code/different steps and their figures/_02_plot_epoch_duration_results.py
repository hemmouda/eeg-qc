from pathlib import Path

import matplotlib.pyplot as plt
from _01_epoch_duration_results import OUTPUT
from core.consts import TARGETS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("_02_epoch_duration_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = False
SAVE_DPI = 300

ANNOTATE_POINTS = True
FONT_SIZE = 13.5

TEXTWIDTH_INCH = 5.455853 * 1.75  # My LaTeX textwidth * scale
FIG_HEIGHT = 0.65 * TEXTWIDTH_INCH  # Play with values between 0.45 and 0.65

COLORS = {
    "class0_any": "#1f77b4",
    "macro_any": "#2ca02c",
    "class1_any": "#ff7f0e",
    "class0_50": "#aec7e8",
    "macro_50": "#98df8a",
    "class1_50": "#ffbb78",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def epoch_sort_key(epoch: str) -> int:
    """'10s' -> 10"""
    return int(epoch[:-1])


def extract_f1_scores(
    results: dict,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """
    Returns:
        epochs,
        class0_f1,
        class1_f1,
        macro_f1
    """
    epochs = sorted(results.keys(), key=epoch_sort_key)

    class0_f1 = [results[epoch]["0"]["f1-score"] for epoch in epochs]
    class1_f1 = [results[epoch]["1"]["f1-score"] for epoch in epochs]
    macro_f1 = [results[epoch]["macro avg"]["f1-score"] for epoch in epochs]

    return epochs, class0_f1, class1_f1, macro_f1


def annotate_curve(
    ax,
    epochs: list[str],
    scores: list[float],
    color: str,
    offset: tuple[int, int],
):
    if not ANNOTATE_POINTS:
        return

    for x, y in zip(epochs, scores):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=FONT_SIZE - 3.5,
            color=color,
        )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_target(target_name: str) -> None:
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_INCH, FIG_HEIGHT))

    # ------------------------------------------------------------------
    # Any overlap
    # ------------------------------------------------------------------

    any_results = OUTPUT["ANY_OVERLAP"][target_name]

    (
        epochs,
        any_class0_f1,
        any_class1_f1,
        any_macro_f1,
    ) = extract_f1_scores(any_results)

    ax.plot(
        epochs,
        any_class0_f1,
        marker="o",
        color=COLORS["class0_any"],
        label="Class 0 F1-score (Any Overlap)",
    )

    ax.plot(
        epochs,
        any_macro_f1,
        marker="o",
        color=COLORS["macro_any"],
        linestyle="--",
        label="Macro F1-score (Any Overlap)",
    )

    ax.plot(
        epochs,
        any_class1_f1,
        marker="o",
        color=COLORS["class1_any"],
        label="Class 1 F1-score (Any Overlap)",
    )

    # ------------------------------------------------------------------
    # 50% overlap
    # ------------------------------------------------------------------

    overlap_results = OUTPUT["50_OVERLAP"][target_name]

    (
        _,
        overlap_class0_f1,
        overlap_class1_f1,
        overlap_macro_f1,
    ) = extract_f1_scores(overlap_results)

    ax.plot(
        epochs,
        overlap_class0_f1,
        marker="x",
        color=COLORS["class0_50"],
        label="Class 0 F1-score (50% Overlap)",
    )

    ax.plot(
        epochs,
        overlap_macro_f1,
        marker="x",
        color=COLORS["macro_50"],
        linestyle="--",
        label="Macro F1-score (50% Overlap)",
    )

    ax.plot(
        epochs,
        overlap_class1_f1,
        marker="x",
        color=COLORS["class1_50"],
        label="Class 1 F1-score (50% Overlap)",
    )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    annotate_curve(
        ax,
        epochs,
        any_class0_f1,
        COLORS["class0_any"],
        (-18, 8),
    )

    annotate_curve(
        ax,
        epochs,
        any_class1_f1,
        COLORS["class1_any"],
        (-18, 8),
    )

    annotate_curve(
        ax,
        epochs,
        any_macro_f1,
        COLORS["macro_any"],
        (-18, 8),
    )

    annotate_curve(
        ax,
        epochs,
        overlap_class0_f1,
        COLORS["class0_50"],
        (18, -8),
    )

    annotate_curve(
        ax,
        epochs,
        overlap_class1_f1,
        COLORS["class1_50"],
        (18, -8),
    )

    annotate_curve(
        ax,
        epochs,
        overlap_macro_f1,
        COLORS["macro_50"],
        (18, -8),
    )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    ax.set_xlabel("Epoch Duration", fontsize=FONT_SIZE)
    ax.set_ylabel("F1-score", fontsize=FONT_SIZE)

    ax.tick_params(axis="both", labelsize=FONT_SIZE)

    handles, labels = ax.get_legend_handles_labels()

    desired_order = [
        "Class 0 F1-score (Any Overlap)",
        "Macro F1-score (Any Overlap)",
        "Class 1 F1-score (Any Overlap)",
        "Class 0 F1-score (50% Overlap)",
        "Macro F1-score (50% Overlap)",
        "Class 1 F1-score (50% Overlap)",
    ]

    handle_map = dict(zip(labels, handles))

    ax.legend(
        [handle_map[label] for label in desired_order],
        desired_order,
        fontsize=FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 1])

    output_path = OUTPUT_DIR / f"epochs_{target_name}_target_f1_scores.png"
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
    for target in TARGETS:
        plot_target(target)


if __name__ == "__main__":
    main()
