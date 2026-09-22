"""Evaluate the trained YOLO detector on the LineMOD validation split."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from src.paths import RUNS_ROOT, YOLO_DATASET_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=RUNS_ROOT / "detect" / "runs" / "linemod_yolo" / "weights" / "best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(YOLO_DATASET_ROOT / "dataset.yaml"),
        split="val",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
