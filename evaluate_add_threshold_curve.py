"""Plot ADD threshold curves for RGB, Depth-only, and RGB-D models."""

import json

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
from src.pose_model import PoseEstimator
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_pose_model import RGBDPoseModel


OBJECT_ID = 1
BATCH_SIZE = 4

OUTPUT_PATH = OUTPUT_ROOT / "add_threshold_curve.json"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "add_threshold_curve.png"


def compute_adds(model, loader, model_points, mode):
    model.eval()
    adds = []

    with torch.no_grad():
        for batch in loader:

            if mode == "rgb":
                rgb, target_t, target_q = batch
                rgb = rgb.to(DEVICE)
                target_t = target_t.to(DEVICE)
                target_q = target_q.to(DEVICE)

                pred_t, pred_q = model(rgb)

            elif mode == "depth":
                _, depth, target_t, target_q = batch
                depth = depth.to(DEVICE)
                target_t = target_t.to(DEVICE)
                target_q = target_q.to(DEVICE)

                pred_t, pred_q = model(depth)

            elif mode == "rgbd":
                rgb, depth, target_t, target_q = batch
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                target_t = target_t.to(DEVICE)
                target_q = target_q.to(DEVICE)

                pred_t, pred_q = model(rgb, depth)

            pred_R = quaternion_to_rotation_matrix(pred_q)
            target_R = quaternion_to_rotation_matrix(target_q)

            for i in range(pred_t.size(0)):
                add = add_metric(
                    model_points,
                    pred_R[i],
                    pred_t[i],
                    target_R[i],
                    target_t[i],
                )

                adds.append(float(add.item()))

    return np.asarray(adds)


def success_curve(adds, thresholds):
    return np.array([
        np.mean(adds < threshold) * 100.0
        for threshold in thresholds
    ])


def main():

    # Datasets
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
        raise ValueError("RGB and RGB-D test samples are not aligned.")

    # DataLoaders
    rgb_loader = DataLoader(
        rgb_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    rgbd_loader = DataLoader(
        rgbd_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 3D model
    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{OBJECT_ID:02d}.ply"
    ).to(DEVICE)

    # Models
    rgb_model = PoseEstimator().to(DEVICE)
    depth_model = DepthPoseModel().to(DEVICE)
    rgbd_model = RGBDPoseModel().to(DEVICE)

    # Checkpoints
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

    # Evaluate
    print("Evaluating RGB...")
    rgb_adds = compute_adds(
        rgb_model,
        rgb_loader,
        model_points,
        "rgb",
    )

    print("Evaluating Depth-only...")
    depth_adds = compute_adds(
        depth_model,
        rgbd_loader,
        model_points,
        "depth",
    )

    print("Evaluating RGB-D...")
    rgbd_adds = compute_adds(
        rgbd_model,
        rgbd_loader,
        model_points,
        "rgbd",
    )

    # Thresholds from 0 to 1000 mm
    thresholds = np.linspace(0, 1000, 201)

    rgb_curve = success_curve(rgb_adds, thresholds)
    depth_curve = success_curve(depth_adds, thresholds)
    rgbd_curve = success_curve(rgbd_adds, thresholds)

    # Save numerical results
    results = {
        "object_id": OBJECT_ID,
        "samples": len(rgb_dataset),
        "threshold_units": "millimetres",
        "thresholds_mm": thresholds.tolist(),
        "rgb_success_percent": rgb_curve.tolist(),
        "depth_success_percent": depth_curve.tolist(),
        "rgbd_success_percent": rgbd_curve.tolist(),
        "mean_add_mm": {
            "rgb": float(np.mean(rgb_adds)),
            "depth": float(np.mean(depth_adds)),
            "rgbd": float(np.mean(rgbd_adds)),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # Plot
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5.5))

    plt.plot(
        thresholds,
        rgb_curve,
        label="RGB",
        linewidth=2,
    )

    plt.plot(
        thresholds,
        depth_curve,
        label="Depth-only",
        linewidth=2,
    )

    plt.plot(
        thresholds,
        rgbd_curve,
        label="RGB-D",
        linewidth=2,
    )

    plt.xlabel("ADD threshold (mm)")
    plt.ylabel("Successful predictions (%)")
    plt.title("ADD Threshold Success Curve")

    plt.xlim(0, 1000)
    plt.ylim(0, 100)

    plt.grid(
        linestyle="--",
        alpha=0.4,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nMean ADD")
    print("=" * 35)
    print(f"RGB:        {np.mean(rgb_adds):.2f} mm")
    print(f"Depth-only: {np.mean(depth_adds):.2f} mm")
    print(f"RGB-D:      {np.mean(rgbd_adds):.2f} mm")
    print("=" * 35)

    print(f"Saved results: {OUTPUT_PATH}")
    print(f"Saved figure:  {FIGURE_PATH}")


if __name__ == "__main__":
    main()