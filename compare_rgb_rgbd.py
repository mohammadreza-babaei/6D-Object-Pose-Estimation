"""Evaluate RGB-only and RGB-D checkpoints on the same LineMOD test split."""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.linemod_dataset import LineMODDataset
from src.model_loader import load_ply_vertices
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.pose_loss import PoseLoss
from src.pose_model import PoseEstimator
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_pose_model import RGBDPoseModel
from src.rgbd_training import evaluate_rgbd
from src.training import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument(
        "--rgb-checkpoint",
        type=Path,
        default=CHECKPOINT_ROOT / "best_model.pth",
    )
    parser.add_argument(
        "--rgbd-checkpoint",
        type=Path,
        default=CHECKPOINT_ROOT / "rgbd_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "rgb_vs_rgbd.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb_dataset = LineMODDataset(str(DATASET_ROOT), args.object_id, "test")
    rgbd_dataset = LineMODRGBDDataset(str(DATASET_ROOT), args.object_id, "test")
    rgb_loader = DataLoader(rgb_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    rgbd_loader = DataLoader(rgbd_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    model_points = load_ply_vertices(DATASET_ROOT / "models" / f"obj_{args.object_id:02d}.ply")
    criterion = PoseLoss()

    rgb_model = PoseEstimator().to(DEVICE)
    load_checkpoint(args.rgb_checkpoint, rgb_model, device=DEVICE)
    rgb_metrics = evaluate(rgb_model, rgb_loader, criterion, model_points)

    rgbd_model = RGBDPoseModel().to(DEVICE)
    load_checkpoint(args.rgbd_checkpoint, rgbd_model, device=DEVICE)
    rgbd_metrics = evaluate_rgbd(rgbd_model, rgbd_loader, criterion, model_points)

    result = {
        "object_id": args.object_id,
        "split": "test",
        "rgb": rgb_metrics,
        "rgbd": rgbd_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
