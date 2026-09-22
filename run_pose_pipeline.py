"""Run YOLO detection followed by RGB or RGB-D pose inference on one sample."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, RUNS_ROOT
from src.pose_model import PoseEstimator
from src.rgbd_pose_model import RGBDPoseModel


RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
DEPTH_RESIZE = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--mode", choices=["rgb", "rgbd"], default="rgb")
    parser.add_argument("--detector", type=Path, default=RUNS_ROOT / "detect" / "runs" / "linemod_yolo" / "weights" / "best.pt")
    parser.add_argument("--pose-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    object_dir = DATASET_ROOT / "data" / f"{args.object_id:02d}"
    rgb_path = object_dir / "rgb" / f"{args.sample_id:04d}.png"
    depth_path = object_dir / "depth" / f"{args.sample_id:04d}.png"
    if args.pose_checkpoint is None:
        args.pose_checkpoint = CHECKPOINT_ROOT / ("rgbd_best.pth" if args.mode == "rgbd" else "best_model.pth")

    image = Image.open(rgb_path).convert("RGB")
    result = YOLO(str(args.detector)).predict(source=str(rgb_path), conf=0.25, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO did not detect an object.")

    best_index = int(result.boxes.conf.argmax().item())
    x1, y1, x2, y2 = result.boxes.xyxy[best_index].cpu().tolist()
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("YOLO returned an invalid bounding box.")

    rgb_crop = image.crop((x1, y1, x2, y2))
    rgb_tensor = RGB_TRANSFORM(rgb_crop).unsqueeze(0).to(DEVICE)

    if args.mode == "rgbd":
        depth = Image.open(depth_path).crop((x1, y1, x2, y2))
        depth_tensor = torch.from_numpy(
            np.array(DEPTH_RESIZE(depth), dtype="float32")
        )
        depth_tensor = (depth_tensor / 1000.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
        model = RGBDPoseModel().to(DEVICE)
    else:
        depth_tensor = None
        model = PoseEstimator().to(DEVICE)

    load_checkpoint(args.pose_checkpoint, model, device=DEVICE)
    model.eval()
    with torch.no_grad():
        if depth_tensor is None:
            translation, quaternion = model(rgb_tensor)
        else:
            translation, quaternion = model(rgb_tensor, depth_tensor)

    print(f"Mode: {args.mode}")
    print(f"Detection confidence: {result.boxes.conf[best_index].item():.4f}")
    print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
    print("Predicted translation (mm):", translation[0].cpu().tolist())
    print("Predicted quaternion [w, x, y, z]:", quaternion[0].cpu().tolist())


if __name__ == "__main__":
    main()
