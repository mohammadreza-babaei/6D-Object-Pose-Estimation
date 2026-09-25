"""Evaluate translation and rotation errors for RGB, Depth-only, and RGB-D models."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.depth_pose_model import DepthPoseModel
from src.linemod_dataset import LineMODDataset
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.pose_model import PoseEstimator
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_pose_model import RGBDPoseModel


OBJECT_ID = 1
BATCH_SIZE = 4

OUTPUT_PATH = OUTPUT_ROOT / "translation_rotation_errors.json"
TRANSLATION_FIGURE = OUTPUT_ROOT / "figures" / "translation_error_comparison.png"
ROTATION_FIGURE = OUTPUT_ROOT / "figures" / "rotation_error_comparison.png"


def quaternion_angular_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return quaternion angular error in degrees."""

    predicted = torch.nn.functional.normalize(predicted, dim=1)
    target = torch.nn.functional.normalize(target, dim=1)

    dot = torch.sum(predicted * target, dim=1).abs()
    dot = torch.clamp(dot, 0.0, 1.0)

    angle_rad = 2.0 * torch.acos(dot)

    return torch.rad2deg(angle_rad)


def evaluate_rgb(model, loader):
    model.eval()

    translation_errors = []
    rotation_errors = []

    with torch.no_grad():
        for images, target_translation, target_rotation in loader:
            images = images.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)

            pred_translation, pred_rotation = model(images)

            translation = torch.linalg.vector_norm(
                pred_translation - target_translation,
                dim=1,
            )

            rotation = quaternion_angular_error(
                pred_rotation,
                target_rotation,
            )

            translation_errors.extend(translation.cpu().tolist())
            rotation_errors.extend(rotation.cpu().tolist())

    return translation_errors, rotation_errors


def evaluate_depth(model, loader):
    model.eval()

    translation_errors = []
    rotation_errors = []

    with torch.no_grad():
        for _, depth, target_translation, target_rotation in loader:
            depth = depth.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)

            pred_translation, pred_rotation = model(depth)

            translation = torch.linalg.vector_norm(
                pred_translation - target_translation,
                dim=1,
            )

            rotation = quaternion_angular_error(
                pred_rotation,
                target_rotation,
            )

            translation_errors.extend(translation.cpu().tolist())
            rotation_errors.extend(rotation.cpu().tolist())

    return translation_errors, rotation_errors


def evaluate_rgbd(model, loader):
    model.eval()

    translation_errors = []
    rotation_errors = []

    with torch.no_grad():
        for rgb, depth, target_translation, target_rotation in loader:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)

            pred_translation, pred_rotation = model(rgb, depth)

            translation = torch.linalg.vector_norm(
                pred_translation - target_translation,
                dim=1,
            )

            rotation = quaternion_angular_error(
                pred_rotation,
                target_rotation,
            )

            translation_errors.extend(translation.cpu().tolist())
            rotation_errors.extend(rotation.cpu().tolist())

    return translation_errors, rotation_errors


def summarize(translation_errors, rotation_errors):
    return {
        "samples": len(translation_errors),
        "translation_error_mm": {
            "mean": float(np.mean(translation_errors)),
            "median": float(np.median(translation_errors)),
            "std": float(np.std(translation_errors)),
        },
        "rotation_error_deg": {
            "mean": float(np.mean(rotation_errors)),
            "median": float(np.median(rotation_errors)),
            "std": float(np.std(rotation_errors)),
        },
    }


def plot_metric(labels, values, ylabel, title, path):
    figure, axis = plt.subplots(figsize=(7.2, 5.2))

    bars = axis.bar(labels, values)

    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.set_axisbelow(True)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")

    plt.close(figure)


def main():
    rgb_dataset = LineMODDataset(
        str(DATASET_ROOT),
        OBJECT_ID,
        "test",
    )

    rgbd_dataset = LineMODRGBDDataset(
        str(DATASET_ROOT),
        OBJECT_ID,
        "test",
    )

    if rgb_dataset.sample_ids != rgbd_dataset.sample_ids:
        raise ValueError("RGB and RGB-D test sample IDs are not aligned.")

    rgb_loader = DataLoader(
        rgb_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    depth_loader = DataLoader(
        rgbd_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    rgbd_loader = DataLoader(
        rgbd_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    rgb_model = PoseEstimator().to(DEVICE)
    depth_model = DepthPoseModel().to(DEVICE)
    rgbd_model = RGBDPoseModel().to(DEVICE)

    load_checkpoint(
        CHECKPOINT_ROOT / "best_model.pth",
        rgb_model,
        device=DEVICE,
    )

    load_checkpoint(
        CHECKPOINT_ROOT / "depth_best.pth",
        depth_model,
        device=DEVICE,
    )

    load_checkpoint(
        CHECKPOINT_ROOT / "rgbd_best.pth",
        rgbd_model,
        device=DEVICE,
    )

    print("Evaluating RGB...")
    rgb_translation, rgb_rotation = evaluate_rgb(
        rgb_model,
        rgb_loader,
    )

    print("Evaluating Depth-only...")
    depth_translation, depth_rotation = evaluate_depth(
        depth_model,
        depth_loader,
    )

    print("Evaluating RGB-D...")
    rgbd_translation, rgbd_rotation = evaluate_rgbd(
        rgbd_model,
        rgbd_loader,
    )

    results = {
        "object_id": OBJECT_ID,
        "split": "test",
        "rgb": summarize(rgb_translation, rgb_rotation),
        "depth": summarize(depth_translation, depth_rotation),
        "rgbd": summarize(rgbd_translation, rgbd_rotation),
        "protocol": {
            "crop": "ground-truth object bounding boxes",
            "translation_metric": "Euclidean distance between predicted and ground-truth translation",
            "translation_units": "millimetres",
            "rotation_metric": "quaternion angular distance",
            "rotation_units": "degrees",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    labels = ["RGB", "Depth-only", "RGB-D"]

    plot_metric(
        labels,
        [
            results["rgb"]["translation_error_mm"]["mean"],
            results["depth"]["translation_error_mm"]["mean"],
            results["rgbd"]["translation_error_mm"]["mean"],
        ],
        "Mean translation error (mm)",
        "Translation Error Comparison",
        TRANSLATION_FIGURE,
    )

    plot_metric(
        labels,
        [
            results["rgb"]["rotation_error_deg"]["mean"],
            results["depth"]["rotation_error_deg"]["mean"],
            results["rgbd"]["rotation_error_deg"]["mean"],
        ],
        "Mean rotation error (degrees)",
        "Rotation Error Comparison",
        ROTATION_FIGURE,
    )

    print(json.dumps(results, indent=2))

    print(f"Saved metrics: {OUTPUT_PATH}")
    print(f"Saved translation plot: {TRANSLATION_FIGURE}")
    print(f"Saved rotation plot: {ROTATION_FIGURE}")


if __name__ == "__main__":
    main()