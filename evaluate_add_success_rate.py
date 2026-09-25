"""Evaluate ADD success rate for RGB, Depth-only, and RGB-D models."""

import json

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

OUTPUT_PATH = OUTPUT_ROOT / "add_success_rate.json"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "add_success_rate.png"


def object_diameter(points):
    """Compute approximate 3D model diameter."""
    points = points.cpu().numpy()

    max_distance = 0.0

    for i in range(len(points)):
        distances = np.linalg.norm(points[i + 1:] - points[i], axis=1)

        if len(distances) > 0:
            max_distance = max(max_distance, float(distances.max()))

    return max_distance


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

    return np.array(adds)


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
        raise ValueError("RGB and RGB-D samples are not aligned.")

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

    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{OBJECT_ID:02d}.ply"
    ).to(DEVICE)

    diameter = object_diameter(model_points)

    threshold = 0.1 * diameter

    print(f"Object diameter: {diameter:.3f} mm")
    print(f"ADD threshold (10% diameter): {threshold:.3f} mm")

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

    results = {}

    for name, adds in [
        ("rgb", rgb_adds),
        ("depth", depth_adds),
        ("rgbd", rgbd_adds),
    ]:

        success = adds < threshold

        results[name] = {
            "mean_add_mm": float(np.mean(adds)),
            "successful_samples": int(success.sum()),
            "total_samples": int(len(adds)),
            "success_rate_percent": float(success.mean() * 100),
        }

    results["protocol"] = {
        "object_diameter_mm": diameter,
        "threshold": "ADD < 10% object diameter",
        "threshold_mm": threshold,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("=" * 40)

    print(
        f"RGB:        {results['rgb']['success_rate_percent']:.2f}%"
    )

    print(
        f"Depth-only: {results['depth']['success_rate_percent']:.2f}%"
    )

    print(
        f"RGB-D:      {results['rgbd']['success_rate_percent']:.2f}%"
    )

    print("=" * 40)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()