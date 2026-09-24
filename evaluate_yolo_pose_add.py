"""Evaluate RGB and RGB-D pose ADD with GT and YOLO-predicted crops."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.geometry import quaternion_to_rotation_matrix
from src.linemod_dataset import LineMODDataset
from src.metrics import add_metric
from src.model_loader import load_ply_vertices
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT, RUNS_ROOT
from src.pose_model import PoseEstimator
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_pose_model import RGBDPoseModel


YOLO_CHECKPOINT = RUNS_ROOT / "detect" / "runs" / "linemod_yolo" / "weights" / "best.pt"
RGB_CHECKPOINT = CHECKPOINT_ROOT / "best_model.pth"
RGBD_CHECKPOINT = CHECKPOINT_ROOT / "rgbd_best.pth"
OUTPUT_PATH = OUTPUT_ROOT / "yolo_pose_add_comparison.json"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "yolo_pose_add_comparison.png"

RGB_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
DEPTH_RESIZE = transforms.Resize(
    (224, 224),
    interpolation=transforms.InterpolationMode.NEAREST,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def clamp_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert xyxy coordinates to the dataset crop convention."""

    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(image_width, int(x2)),
        min(image_height, int(y2)),
    )


def valid_box(box: tuple[int, int, int, int]) -> bool:
    return box[2] > box[0] and box[3] > box[1]


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Calculate IoU for two integer xyxy boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def gt_box(dataset: LineMODDataset, sample_id: int, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Read and clamp the object bbox using the dataset loader's convention."""

    annotation = next(
        item
        for item in dataset.ground_truth[sample_id]
        if int(item["obj_id"]) == dataset.object_id
    )
    x, y, width, height = annotation["obj_bb"]
    return clamp_box((x, y, x + width, y + height), image_size)


def best_class_zero_box(result) -> tuple[float, float, float, float] | None:
    """Return the highest-confidence class-0 YOLO prediction."""

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    class_mask = boxes.cls == 0
    if not bool(class_mask.any()):
        return None
    candidate_indices = torch.where(class_mask)[0]
    best_offset = int(boxes.conf[candidate_indices].argmax().item())
    best_index = int(candidate_indices[best_offset].item())
    return tuple(float(value) for value in boxes.xyxy[best_index].cpu().tolist())


def predicted_inputs(
    rgb_path: Path,
    depth_path: Path,
    crop: tuple[int, int, int, int],
    depth_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create RGB-D tensors from one shared crop and existing preprocessing."""

    rgb = Image.open(rgb_path).convert("RGB").crop(crop)
    depth = Image.open(depth_path).crop(crop)
    rgb_tensor = RGB_TRANSFORM(rgb)
    depth_tensor = torch.from_numpy(
        np.array(DEPTH_RESIZE(depth), dtype=np.float32)
    )
    depth_tensor = depth_tensor.mul(depth_scale / 1000.0).unsqueeze(0)
    return rgb_tensor, depth_tensor


def pose_add(
    model_points: torch.Tensor,
    predicted_translation: torch.Tensor,
    predicted_rotation: torch.Tensor,
    target_translation: torch.Tensor,
    target_rotation: torch.Tensor,
) -> float:
    """Calculate ADD using the repository's existing implementation."""

    predicted_matrix = quaternion_to_rotation_matrix(predicted_rotation.unsqueeze(0))[0]
    target_matrix = quaternion_to_rotation_matrix(target_rotation.unsqueeze(0))[0]
    return float(
        add_metric(
            model_points,
            predicted_matrix,
            predicted_translation,
            target_matrix,
            target_translation,
        ).item()
    )


def run_pose_models(
    rgb_model: PoseEstimator,
    rgbd_model: RGBDPoseModel,
    rgb_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    model_points: torch.Tensor,
    target_translation: torch.Tensor,
    target_rotation: torch.Tensor,
) -> dict[str, float]:
    """Run both pose models for one preprocessed crop."""

    with torch.no_grad():
        rgb_translation, rgb_rotation = rgb_model(rgb_tensor.unsqueeze(0).to(DEVICE))
        rgbd_translation, rgbd_rotation = rgbd_model(
            rgb_tensor.unsqueeze(0).to(DEVICE),
            depth_tensor.unsqueeze(0).to(DEVICE),
        )
    return {
        "rgb_add_mm": pose_add(
            model_points,
            rgb_translation[0].cpu(),
            rgb_rotation[0].cpu(),
            target_translation,
            target_rotation,
        ),
        "rgbd_add_mm": pose_add(
            model_points,
            rgbd_translation[0].cpu(),
            rgbd_rotation[0].cpu(),
            target_translation,
            target_rotation,
        ),
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"valid_samples": 0, "mean_add_mm": None, "median_add_mm": None}
    return {
        "valid_samples": len(values),
        "mean_add_mm": float(np.mean(values)),
        "median_add_mm": float(np.median(values)),
    }


def draw_examples(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save representative bbox and per-sample ADD comparisons."""

    selected = records[:1] if len(records) == 1 else [records[0], records[len(records) // 2], records[-1]]
    figure, axes = plt.subplots(
        1,
        len(selected),
        figsize=(6.5 * len(selected), 5.8),
        squeeze=False,
    )
    for axis, record in zip(axes[0], selected):
        image = Image.open(record["rgb_path"]).convert("RGB")
        axis.imshow(image)
        gt = record["gt_bbox"]
        axis.add_patch(
            Rectangle(
                (gt[0], gt[1]),
                gt[2] - gt[0],
                gt[3] - gt[1],
                fill=False,
                edgecolor="#00C853",
                linewidth=2.5,
            )
        )
        predicted = record["yolo_bbox"]
        if predicted is not None:
            axis.add_patch(
                Rectangle(
                    (predicted[0], predicted[1]),
                    predicted[2] - predicted[0],
                    predicted[3] - predicted[1],
                    fill=False,
                    edgecolor="#FF1744",
                    linewidth=2.5,
                )
            )
        lines = [
            f"IoU: {record['bbox_iou']:.3f}" if record["bbox_iou"] is not None else "IoU: unavailable",
            f"RGB GT ADD: {record['rgb_gt_add_mm']:.1f} mm",
            f"RGB YOLO ADD: {record['rgb_yolo_add_mm']:.1f} mm" if record["rgb_yolo_add_mm"] is not None else "RGB YOLO ADD: unavailable",
            f"RGB-D GT ADD: {record['rgbd_gt_add_mm']:.1f} mm",
            f"RGB-D YOLO ADD: {record['rgbd_yolo_add_mm']:.1f} mm" if record["rgbd_yolo_add_mm"] is not None else "RGB-D YOLO ADD: unavailable",
        ]
        axis.set_title(f"Sample {record['sample_id']:04d}\n" + "\n".join(lines), fontsize=9)
        axis.axis("off")

    figure.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#00C853", label="Ground-truth bbox"),
            Patch(facecolor="none", edgecolor="#FF1744", label="YOLO bbox"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle("End-to-end YOLO crop impact on pose ADD", fontsize=16)
    figure.tight_layout(rect=(0, 0.08, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be at least 1")

    rgb_dataset = LineMODDataset(str(DATASET_ROOT), args.object_id, "test")
    rgbd_dataset = LineMODRGBDDataset(str(DATASET_ROOT), args.object_id, "test")
    if rgb_dataset.sample_ids != rgbd_dataset.sample_ids:
        raise ValueError("RGB and RGB-D test sample IDs are not aligned.")

    sample_ids = rgb_dataset.sample_ids[: args.limit]
    yolo_model = YOLO(str(YOLO_CHECKPOINT))
    rgb_model = PoseEstimator().to(DEVICE).eval()
    rgbd_model = RGBDPoseModel().to(DEVICE).eval()
    load_checkpoint(RGB_CHECKPOINT, rgb_model, device=DEVICE)
    load_checkpoint(RGBD_CHECKPOINT, rgbd_model, device=DEVICE)
    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{args.object_id:02d}.ply"
    )

    rgb_gt_adds: list[float] = []
    rgb_yolo_adds: list[float] = []
    rgbd_gt_adds: list[float] = []
    rgbd_yolo_adds: list[float] = []
    records: list[dict[str, Any]] = []
    missing_yolo: list[int] = []

    for sample_id_text in sample_ids:
        sample_id = int(sample_id_text)
        rgb_path = DATASET_ROOT / "data" / f"{args.object_id:02d}" / "rgb" / f"{sample_id:04d}.png"
        depth_path = DATASET_ROOT / "data" / f"{args.object_id:02d}" / "depth" / f"{sample_id:04d}.png"
        with Image.open(rgb_path) as image:
            image_size = image.size
        gt_crop = gt_box(rgb_dataset, sample_id, image_size)
        if not valid_box(gt_crop):
            raise ValueError(f"Invalid ground-truth crop for sample {sample_id}")

        rgb_gt, depth_gt, target_translation, target_rotation = rgbd_dataset[
            rgbd_dataset.sample_ids.index(sample_id_text)
        ]
        gt_results = run_pose_models(
            rgb_model,
            rgbd_model,
            rgb_gt,
            depth_gt,
            model_points,
            target_translation,
            target_rotation,
        )
        rgb_gt_adds.append(gt_results["rgb_add_mm"])
        rgbd_gt_adds.append(gt_results["rgbd_add_mm"])

        detection = yolo_model.predict(
            source=str(rgb_path),
            conf=args.confidence,
            imgsz=args.image_size,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
        raw_prediction = best_class_zero_box(detection)
        yolo_crop = clamp_box(raw_prediction, image_size) if raw_prediction else None
        yolo_valid = yolo_crop is not None and valid_box(yolo_crop)
        if not yolo_valid:
            missing_yolo.append(sample_id)
            yolo_crop = None
            rgb_yolo_add = None
            rgbd_yolo_add = None
            bbox_iou = None
        else:
            rgb_yolo, depth_yolo = predicted_inputs(
                rgb_path,
                depth_path,
                yolo_crop,
                float(rgbd_dataset.info[sample_id].get("depth_scale", 1.0)),
            )
            yolo_results = run_pose_models(
                rgb_model,
                rgbd_model,
                rgb_yolo,
                depth_yolo,
                model_points,
                target_translation,
                target_rotation,
            )
            rgb_yolo_add = yolo_results["rgb_add_mm"]
            rgbd_yolo_add = yolo_results["rgbd_add_mm"]
            rgb_yolo_adds.append(rgb_yolo_add)
            rgbd_yolo_adds.append(rgbd_yolo_add)
            bbox_iou = box_iou(gt_crop, yolo_crop)

        records.append(
            {
                "sample_id": sample_id,
                "rgb_path": str(rgb_path),
                "gt_bbox": gt_crop,
                "yolo_bbox": yolo_crop,
                "bbox_iou": bbox_iou,
                "rgb_gt_add_mm": gt_results["rgb_add_mm"],
                "rgb_yolo_add_mm": rgb_yolo_add,
                "rgbd_gt_add_mm": gt_results["rgbd_add_mm"],
                "rgbd_yolo_add_mm": rgbd_yolo_add,
            }
        )

    result = {
        "object_id": args.object_id,
        "split": "test",
        "evaluated_samples": len(sample_ids),
        "yolo_detections": {
            "successful": len(sample_ids) - len(missing_yolo),
            "missing": len(missing_yolo),
            "missing_sample_ids": missing_yolo,
            "selection": "highest-confidence class-0 prediction",
        },
        "conditions": {
            "rgb_ground_truth_bbox": summarize(rgb_gt_adds),
            "rgb_yolo_bbox": summarize(rgb_yolo_adds),
            "rgbd_ground_truth_bbox": summarize(rgbd_gt_adds),
            "rgbd_yolo_bbox": summarize(rgbd_yolo_adds),
        },
        "checkpoints": {
            "yolo": str(YOLO_CHECKPOINT),
            "rgb": str(RGB_CHECKPOINT),
            "rgbd": str(RGBD_CHECKPOINT),
        },
        "device": str(DEVICE),
        "preprocessing": {
            "source_image_size": "original 640x480 RGB/depth images",
            "crop_coordinates": "integer xyxy coordinates clamped to image bounds",
            "rgb": "crop, resize to 224x224, ToTensor, ImageNet normalization",
            "depth": "same crop as RGB, nearest-neighbour resize to 224x224, depth_scale / 1000 to metres",
            "ground_truth_branch": "dataset obj_bb crop",
            "yolo_branch": "highest-confidence class-0 YOLO crop",
        },
        "add": {
            "implementation": "src.metrics.add_metric",
            "model_points": f"data/linemod/Linemod_preprocessed/models/obj_{args.object_id:02d}.ply",
            "units": "millimetres",
            "description": "Mean distance between corresponding PLY model points transformed by predicted and ground-truth poses",
        },
        "per_sample": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    draw_examples(records, args.figure)
    print(json.dumps(result["conditions"], indent=2))
    print(f"Saved comparison: {args.output}")
    print(f"Saved visualization: {args.figure}")


if __name__ == "__main__":
    main()
