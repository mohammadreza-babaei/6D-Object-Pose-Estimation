"""Plot RGB versus RGB-D ADD results from the saved experiment JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "rgb_vs_rgbd.json"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "rgb_vs_rgbd_add.png"


def main() -> None:
    """Create and save a publication-ready ADD comparison chart."""

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    methods = ["RGB", "RGB-D"]
    add_values = [
        float(results["rgb"]["add"]),
        float(results["rgbd"]["add"]),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
        }
    )

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    bars = axis.bar(
        methods,
        add_values,
        color=["#4472C4", "#70AD47"],
        width=0.58,
        edgecolor="#333333",
        linewidth=0.8,
    )

    axis.set_title("RGB vs. RGB-D 6D Pose Estimation")
    axis.set_xlabel("Input modality")
    axis.set_ylabel("ADD (mm)")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    upper_limit = max(add_values) * 1.18 if max(add_values) > 0 else 1.0
    axis.set_ylim(0, upper_limit)

    for bar, value in zip(bars, add_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + upper_limit * 0.018,
            f"{value:.2f} mm",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_PATH,
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"Saved ADD comparison chart: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
