"""Save and plot epoch-by-epoch pose training history."""

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "train_translation_loss",
    "train_rotation_loss",
    "validation_translation_loss",
    "validation_rotation_loss",
    "validation_add",
]


def save_training_history(
    history: Iterable[dict[str, float]],
    path: str | Path,
) -> None:
    """Write training history rows to a CSV file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(history)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_history(
    history: Iterable[dict[str, float]],
    output_dir: str | Path,
    prefix: str,
) -> None:
    """Save reusable paper-quality plots for a pose training history."""

    rows = list(history)
    epochs = [row["epoch"] for row in rows]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "loss",
            "Total loss",
            "Loss",
            [("train_loss", "Train"), ("validation_loss", "Validation")],
        ),
        (
            "translation_loss",
            "Translation loss",
            "MSE loss",
            [("train_translation_loss", "Train"), ("validation_translation_loss", "Validation")],
        ),
        (
            "rotation_loss",
            "Rotation loss",
            "Sign-invariant quaternion loss",
            [("train_rotation_loss", "Train"), ("validation_rotation_loss", "Validation")],
        ),
    ]

    for name, title, ylabel, series in plots:
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for field, label in series:
            axis.plot(epochs, [row[field] for row in rows], marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path / f"{prefix}_{name}.png", dpi=300)
        plt.close(figure)

    if rows and rows[0].get("validation_add") is not None:
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        axis.plot(
            epochs,
            [row["validation_add"] for row in rows],
            marker="o",
            color="#70AD47",
        )
        axis.set_title("Validation ADD")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("ADD (mm)")
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        figure.tight_layout()
        figure.savefig(output_path / f"{prefix}_validation_add.png", dpi=300)
        plt.close(figure)