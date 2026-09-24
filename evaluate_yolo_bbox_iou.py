"""Evaluate YOLO bounding-box IoU against the LineMOD validation labels."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from ultralytics import YOLO

from src.paths import OUTPUT_ROOT, RUNS_ROOT, YOLO_DATASET_ROOT


MODEL_PATH = RUNS_ROOT / "detect" / "runs" / "linemod_yolo" / "weights" / "best.pt"
IMAGE_DIR = YOLO_DATASET_ROOT / "images" / "val"
LABEL_DIR = YOLO_DATASET_ROOT / "labels" / "val"
METRICS_PATH = OUTPUT_ROOT / "yolo_bbox_iou_metrics.json"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "yolo_bbox_iou_examples.png"


def yolo_label_to_xyxy(label_path: Path, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Convert the single normalized YOLO label to image-coordinate xyxy."""

    image_width, image_height = image_size
    values = label_path.read_text(encoding="utf-8").strip().split()
    if len(values) != 5:
        raise ValueError(f"Expected one YOLO label in {label_path}")

    _, x_center, y_center, box_width, box_height = map(float, values)
    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height
    return (
        x_center - box_width / 2,
        y_center - box_height / 2,
        x_center + box_width / 2,
        y_center + box_height / 2,
    )


def intersection_over_union(
    predicted: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    """Calculate IoU for two xyxy boxes."""

    intersection_left = max(predicted[0], target[0])
    intersection_top = max(predicted[1], target[1])
    intersection_right = min(predicted[2], target[2])
    intersection_bottom = min(predicted[3], target[3])
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    predicted_area = max(0.0, predicted[2] - predicted[0]) * max(
        0.0, predicted[3] - predicted[1]
    )
    target_area = max(0.0, target[2] - target[0]) * max(
        0.0, target[3] - target[1]
    )
    union_area = predicted_area + target_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def best_prediction(result) -> tuple[float, float, float, float] | None:
    """Select the highest-confidence prediction for the single object class."""

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


def draw_examples(records: list[dict], output_path: Path) -> None:
    """Save representative predicted-vs-ground-truth box examples."""

    selected = [records[0], records[len(records) // 2], records[-1]]
    selected = list({record["image_name"]: record for record in selected}.values())
    columns = len(selected)
    figure, axes = plt.subplots(1, columns, figsize=(6.5 * columns, 5.5), squeeze=False)

    for axis, record in zip(axes[0], selected):
        image = Image.open(record["image_path"]).convert("RGB")
        axis.imshow(image)
        ground_truth = record["ground_truth"]
        axis.add_patch(
            Rectangle(
                (ground_truth[0], ground_truth[1]),
                ground_truth[2] - ground_truth[0],
                ground_truth[3] - ground_truth[1],
                fill=False,
                edgecolor="#00C853",
                linewidth=2.5,
            )
        )
        predicted = record["prediction"]
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
        axis.set_title(f"{record['image_name']} | IoU: {record['iou']:.3f}")
        axis.axis("off")

    figure.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#00C853", label="Ground truth"),
            Patch(facecolor="none", edgecolor="#FF1744", label="Prediction"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle("YOLO predicted versus ground-truth bounding boxes", fontsize=16)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=METRICS_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No validation images found in {IMAGE_DIR}")

    model = YOLO(str(args.model))
    records = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth = yolo_label_to_xyxy(
            LABEL_DIR / f"{image_path.stem}.txt",
            image_size,
        )
        result = model.predict(
            source=str(image_path),
            conf=args.confidence,
            imgsz=args.image_size,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )[0]
        prediction = best_prediction(result)
        iou = intersection_over_union(prediction, ground_truth) if prediction else 0.0
        records.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "iou": iou,
                "detected": prediction is not None,
            }
        )

    ious = [record["iou"] for record in records]
    sorted_ious = sorted(ious)
    middle = len(sorted_ious) // 2
    median_iou = (
        sorted_ious[middle]
        if len(sorted_ious) % 2
        else (sorted_ious[middle - 1] + sorted_ious[middle]) / 2
    )
    metrics = {
        "model": str(args.model),
        "split": "val",
        "image_size": args.image_size,
        "confidence_threshold": args.confidence,
        "matching": "single object-01 GT box matched with highest-confidence class-0 prediction per image; missing predictions receive IoU 0",
        "evaluated_images": len(records),
        "evaluated_objects": len(records),
        "detected_objects": sum(record["detected"] for record in records),
        "missing_predictions": sum(not record["detected"] for record in records),
        "mean_iou": sum(ious) / len(ious),
        "median_iou": median_iou,
        "percentage_iou_at_least_0_5": 100.0 * sum(iou >= 0.5 for iou in ious) / len(ious),
        "iou_min": min(ious),
        "iou_max": max(ious),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    draw_examples(records, args.figure)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {args.output}")
    print(f"Saved visualization: {args.figure}")


if __name__ == "__main__":
    main()