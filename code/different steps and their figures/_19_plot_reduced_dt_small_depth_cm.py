from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _18_reduced_dt_small_depth_cm import OUTPUT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("_19_plot_reduced_dt_small_depth_cm_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SHOW_PLOTS = True
SAVE_DPI = 300

FONT_SIZE = 12.5

TEXTWIDTH_INCH = 5.455853 * 1.275
FIG_HEIGHT = TEXTWIDTH_INCH / 1.275

TARGET_LABELS = {
    "artf": "artf_ss",
    "musc": "musc_ss",
    "eyem": "eyem_ss",
    "elec": "elec_ss",
}

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------


def plot_target(target: str):
    vals = OUTPUT[target]

    cm = (
        np.array(
            [
                [vals["tn"], vals["fp"]],
                [vals["fn"], vals["tp"]],
            ]
        )
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(TEXTWIDTH_INCH, FIG_HEIGHT),
    )

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    ticks = [0, 20, 40, 60, 80, 100]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t}%" for t in ticks])

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Predicted 0", "Predicted 1"],
        fontsize=FONT_SIZE,
    )

    ax.set_yticklabels(
        ["True 0", "True 1"],
        fontsize=FONT_SIZE,
    )

    ax.set_xlabel("Predicted label", fontsize=FONT_SIZE)
    ax.set_ylabel("True label", fontsize=FONT_SIZE)

    threshold = 50.0

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}%",
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{TARGET_LABELS.get(target)}_confusion_matrix.png"

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
    for target in OUTPUT:
        plot_target(target)


if __name__ == "__main__":
    main()
