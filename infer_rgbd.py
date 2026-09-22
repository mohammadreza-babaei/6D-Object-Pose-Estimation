"""Run RGB-D pose inference for one LineMOD sample."""

import argparse
from pathlib import Path

import torch

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_pose_model import RGBDPoseModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_ROOT / "rgbd_best.pth")
    args = parser.parse_args()

    dataset = LineMODRGBDDataset(str(DATASET_ROOT), args.object_id, "test")
    sample_index = next(
        index
        for index, sample_id in enumerate(dataset.sample_ids)
        if int(sample_id) == args.sample_id
    )
    rgb, depth, translation, rotation = dataset[sample_index]

    model = RGBDPoseModel().to(DEVICE).eval()
    load_checkpoint(args.checkpoint, model, device=DEVICE)
    with torch.no_grad():
        predicted_translation, predicted_rotation = model(
            rgb.unsqueeze(0).to(DEVICE), depth.unsqueeze(0).to(DEVICE)
        )

    print(f"Sample: {args.sample_id:04d}")
    print(f"Device: {DEVICE}")
    print("Predicted translation (mm):", predicted_translation[0].cpu().tolist())
    print("Predicted quaternion [w, x, y, z]:", predicted_rotation[0].cpu().tolist())
    print("Ground-truth translation (mm):", translation.tolist())
    print("Ground-truth quaternion [w, x, y, z]:", rotation.tolist())


if __name__ == "__main__":
    main()
