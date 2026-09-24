"""Compare RGB-only, Depth-only, and RGB-D pose models on GT crops."""

import argparse
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
from src.geometry import quaternion_to_rotation_matrix
from src.linemod_dataset import LineMODDataset
from src.metrics import add_metric
from src.model_loader import load_ply_vertices
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.pose_loss import PoseLoss
from src.pose_model import PoseEstimator
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_pose_model import RGBDPoseModel


DEPTH_CHECKPOINT = CHECKPOINT_ROOT / "depth_best.pth"
OUTPUT_PATH = OUTPUT_ROOT / "rgb_depth_rgbd_ablation.json"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "rgb_depth_rgbd_add.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--depth-checkpoint", type=Path, default=DEPTH_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    return parser.parse_args()


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def evaluate_rgb(
    model: PoseEstimator,
    loader,
    criterion: PoseLoss,
    model_points: torch.Tensor,
) -> tuple[dict[str, float], list[float]]:
    model.eval()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    adds = []
    samples = 0
    with torch.no_grad():
        for images, target_translation, target_rotation in loader:
            images = images.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)
            pred_translation, pred_rotation = model(images)
            loss, translation_loss, rotation_loss = criterion(
                pred_translation,
                pred_rotation,
                target_translation,
                target_rotation,
            )
            batch_size = images.size(0)
            samples += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["translation_loss"] += translation_loss.item() * batch_size
            totals["rotation_loss"] += rotation_loss.item() * batch_size
            pred_matrices = quaternion_to_rotation_matrix(pred_rotation)
            target_matrices = quaternion_to_rotation_matrix(target_rotation)
            for index in range(batch_size):
                adds.append(
                    float(
                        add_metric(
                            model_points,
                            pred_matrices[index],
                            pred_translation[index],
                            target_matrices[index],
                            target_translation[index],
                        ).item()
                    )
                )
    metrics = {name: value / samples for name, value in totals.items()}
    metrics["add"] = float(np.mean(adds))
    metrics["median_add"] = float(np.median(adds))
    metrics["samples"] = samples
    return metrics, adds


def evaluate_depth(
    model: DepthPoseModel,
    loader,
    criterion: PoseLoss,
    model_points: torch.Tensor,
) -> tuple[dict[str, float], list[float]]:
    model.eval()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    adds = []
    samples = 0
    with torch.no_grad():
        for _, depth, target_translation, target_rotation in loader:
            depth = depth.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)
            pred_translation, pred_rotation = model(depth)
            loss, translation_loss, rotation_loss = criterion(
                pred_translation,
                pred_rotation,
                target_translation,
                target_rotation,
            )
            batch_size = depth.size(0)
            samples += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["translation_loss"] += translation_loss.item() * batch_size
            totals["rotation_loss"] += rotation_loss.item() * batch_size
            pred_matrices = quaternion_to_rotation_matrix(pred_rotation)
            target_matrices = quaternion_to_rotation_matrix(target_rotation)
            for index in range(batch_size):
                adds.append(
                    float(
                        add_metric(
                            model_points,
                            pred_matrices[index],
                            pred_translation[index],
                            target_matrices[index],
                            target_translation[index],
                        ).item()
                    )
                )
    metrics = {name: value / samples for name, value in totals.items()}
    metrics["add"] = float(np.mean(adds))
    metrics["median_add"] = float(np.median(adds))
    metrics["samples"] = samples
    return metrics, adds


def evaluate_rgbd_batches(
    model: RGBDPoseModel,
    loader,
    criterion: PoseLoss,
    model_points: torch.Tensor,
) -> tuple[dict[str, float], list[float]]:
    model.eval()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    adds = []
    samples = 0
    with torch.no_grad():
        for rgb, depth, target_translation, target_rotation in loader:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)
            pred_translation, pred_rotation = model(rgb, depth)
            loss, translation_loss, rotation_loss = criterion(
                pred_translation,
                pred_rotation,
                target_translation,
                target_rotation,
            )
            batch_size = rgb.size(0)
            samples += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["translation_loss"] += translation_loss.item() * batch_size
            totals["rotation_loss"] += rotation_loss.item() * batch_size
            pred_matrices = quaternion_to_rotation_matrix(pred_rotation)
            target_matrices = quaternion_to_rotation_matrix(target_rotation)
            for index in range(batch_size):
                adds.append(
                    float(
                        add_metric(
                            model_points,
                            pred_matrices[index],
                            pred_translation[index],
                            target_matrices[index],
                            target_translation[index],
                        ).item()
                    )
                )
    metrics = {name: value / samples for name, value in totals.items()}
    metrics["add"] = float(np.mean(adds))
    metrics["median_add"] = float(np.median(adds))
    metrics["samples"] = samples
    return metrics, adds


def plot_results(results: dict, output_path: Path) -> None:
    labels = ["RGB", "Depth-only", "RGB-D"]
    values = [
        results["rgb"]["add"],
        results["depth"]["add"],
        results["rgbd"]["add"],
    ]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    bars = axis.bar(labels, values, color=["#4472C4", "#ED7D31", "#70AD47"])
    axis.set_ylabel("Mean ADD (mm)")
    axis.set_title("RGB vs. Depth-only vs. RGB-D pose estimation")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rgb_dataset = LineMODDataset(str(DATASET_ROOT), args.object_id, "test")
    rgbd_dataset = LineMODRGBDDataset(str(DATASET_ROOT), args.object_id, "test")
    if rgb_dataset.sample_ids != rgbd_dataset.sample_ids:
        raise ValueError("RGB and RGB-D test sample IDs are not aligned.")

    rgb_loader = DataLoader(
        rgb_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    depth_loader = DataLoader(
        rgbd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    rgbd_loader = DataLoader(
        rgbd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{args.object_id:02d}.ply"
    ).to(DEVICE)
    criterion = PoseLoss()

    rgb_model = PoseEstimator().to(DEVICE)
    depth_model = DepthPoseModel().to(DEVICE)
    rgbd_model = RGBDPoseModel().to(DEVICE)
    load_checkpoint(CHECKPOINT_ROOT / "best_model.pth", rgb_model, device=DEVICE)
    load_checkpoint(args.depth_checkpoint, depth_model, device=DEVICE)
    load_checkpoint(CHECKPOINT_ROOT / "rgbd_best.pth", rgbd_model, device=DEVICE)

    rgb_metrics, _ = evaluate_rgb(rgb_model, rgb_loader, criterion, model_points)
    depth_metrics, _ = evaluate_depth(depth_model, depth_loader, criterion, model_points)
    rgbd_metrics, _ = evaluate_rgbd_batches(rgbd_model, rgbd_loader, criterion, model_points)
    results = {
        "object_id": args.object_id,
        "split": "test",
        "sample_alignment": {
            "rgb_samples": len(rgb_dataset),
            "depth_samples": len(rgbd_dataset),
            "same_sample_ids": True,
        },
        "rgb": rgb_metrics,
        "depth": depth_metrics,
        "rgbd": rgbd_metrics,
        "checkpoints": {
            "rgb": str(CHECKPOINT_ROOT / "best_model.pth"),
            "depth": str(args.depth_checkpoint),
            "rgbd": str(CHECKPOINT_ROOT / "rgbd_best.pth"),
        },
        "parameter_counts": {
            "rgb": parameter_counts(rgb_model),
            "depth": parameter_counts(depth_model),
            "rgbd": parameter_counts(rgbd_model),
        },
        "protocol": {
            "crop": "ground-truth object bounding boxes",
            "rgb_input": "224x224 with ImageNet normalization",
            "depth_input": "224x224 nearest-neighbour resize with existing depth_scale handling",
            "add_units": "millimetres",
            "add_implementation": "src.metrics.add_metric",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, args.figure)
    print(json.dumps(results, indent=2))
    print(f"Saved comparison: {args.output}")
    print(f"Saved plot: {args.figure}")


if __name__ == "__main__":
    main()
