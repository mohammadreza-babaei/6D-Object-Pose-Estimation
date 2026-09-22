"""Visualize RGB and RGB-D pose predictions on LineMOD test samples."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.linemod_dataset import LineMODDataset
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.rgbd_dataset import LineMODRGBDDataset
from src.pose_model import PoseEstimator
from src.rgbd_pose_model import RGBDPoseModel


OUTPUT_PATH = OUTPUT_ROOT / "pose_prediction_examples.png"
RGB_CHECKPOINT = CHECKPOINT_ROOT / "best_model.pth"
RGBD_CHECKPOINT = CHECKPOINT_ROOT / "rgbd_best.pth"


def quaternion_angular_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Return the sign-invariant quaternion error in degrees."""

    predicted = torch.nn.functional.normalize(predicted, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    dot_product = torch.abs(torch.sum(predicted * target, dim=-1))
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle = 2.0 * torch.acos(dot_product)
    return float(torch.rad2deg(angle).item())


def translation_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Return Euclidean translation error in millimetres."""

    return float(torch.linalg.vector_norm(predicted - target).item())


def image_for_display(
    dataset: LineMODDataset,
    index: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Load the full RGB frame and its ground-truth object box."""

    sample_id = int(dataset.sample_ids[index])
    image_path = dataset.object_dir / "rgb" / f"{sample_id:04d}.png"
    image = Image.open(image_path).convert("RGB")
    annotation = next(
        item
        for item in dataset.ground_truth[sample_id]
        if int(item["obj_id"]) == dataset.object_id
    )

    x, y, width, height = annotation["obj_bb"]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image.width, int(x + width))
    y2 = min(image.height, int(y + height))
    return image, (x1, y1, x2, y2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num-samples must be at least 1")

    rgb_dataset = LineMODDataset(
        str(DATASET_ROOT), args.object_id, split="test"
    )
    rgbd_dataset = LineMODRGBDDataset(
        str(DATASET_ROOT), args.object_id, split="test"
    )
    if rgb_dataset.sample_ids != rgbd_dataset.sample_ids:
        raise ValueError("RGB and RGB-D test splits are not aligned.")
    if args.start_index < 0 or args.start_index + args.num_samples > len(rgb_dataset):
        raise ValueError("Requested sample range is outside the test split.")

    rgb_model = PoseEstimator().to(DEVICE).eval()
    rgbd_model = RGBDPoseModel().to(DEVICE).eval()
    load_checkpoint(RGB_CHECKPOINT, rgb_model, device=DEVICE)
    load_checkpoint(RGBD_CHECKPOINT, rgbd_model, device=DEVICE)

    predictions = []
    with torch.no_grad():
        for index in range(args.start_index, args.start_index + args.num_samples):
            rgb_image, target_translation, target_quaternion = rgb_dataset[index]
            rgbd_image, depth, rgbd_translation, rgbd_quaternion = rgbd_dataset[index]

            if not torch.equal(target_translation, rgbd_translation):
                raise ValueError("RGB and RGB-D translation targets differ.")
            if not torch.equal(target_quaternion, rgbd_quaternion):
                raise ValueError("RGB and RGB-D rotation targets differ.")

            display_image, display_bbox = image_for_display(rgb_dataset, index)
            rgb_translation, rgb_quaternion = rgb_model(
                rgb_image.unsqueeze(0).to(DEVICE)
            )
            rgbd_translation, rgbd_quaternion = rgbd_model(
                rgbd_image.unsqueeze(0).to(DEVICE),
                depth.unsqueeze(0).to(DEVICE),
            )

            predictions.append(
                {
                    "sample_id": int(rgb_dataset.sample_ids[index]),
                    "image": display_image,
                    "bbox": display_bbox,
                    "target_translation": target_translation,
                    "target_quaternion": target_quaternion,
                    "rgb_translation": rgb_translation[0].cpu(),
                    "rgb_quaternion": rgb_quaternion[0].cpu(),
                    "rgbd_translation": rgbd_translation[0].cpu(),
                    "rgbd_quaternion": rgbd_quaternion[0].cpu(),
                }
            )

    columns = 2
    rows = (len(predictions) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 5.8 * rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, prediction in zip(axes_flat, predictions):
        target_translation = prediction["target_translation"]
        target_quaternion = prediction["target_quaternion"]
        rgb_translation = prediction["rgb_translation"]
        rgb_quaternion = prediction["rgb_quaternion"]
        rgbd_translation = prediction["rgbd_translation"]
        rgbd_quaternion = prediction["rgbd_quaternion"]

        rgb_translation_error = translation_error(
            rgb_translation, target_translation
        )
        rgbd_translation_error = translation_error(
            rgbd_translation, target_translation
        )
        rgb_rotation_error = quaternion_angular_error(
            rgb_quaternion, target_quaternion
        )
        rgbd_rotation_error = quaternion_angular_error(
            rgbd_quaternion, target_quaternion
        )

        axis.imshow(prediction["image"])
        x1, y1, x2, y2 = prediction["bbox"]
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#00C853",
                linewidth=2.0,
            )
        )
        axis.set_title(f"LineMOD object 1 | test sample {prediction['sample_id']:04d}")
        axis.axis("off")
        axis.text(
            0.02,
            -0.08,
            "\n".join(
                [
                    "Ground truth t (mm): " + format_vector(target_translation),
                    "RGB t (mm):          " + format_vector(rgb_translation),
                    f"RGB translation error:   {rgb_translation_error:.2f} mm",
                    "RGB-D t (mm):        " + format_vector(rgbd_translation),
                    f"RGB-D translation error: {rgbd_translation_error:.2f} mm",
                    f"Quaternion error RGB / RGB-D: {rgb_rotation_error:.2f} / {rgbd_rotation_error:.2f} deg",
                ]
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
        )

    for axis in axes_flat[len(predictions):]:
        axis.axis("off")

    figure.suptitle(
        "Qualitative RGB and RGB-D 6D Pose Predictions",
        fontsize=17,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=0.04,
        top=0.94,
        wspace=0.08,
        hspace=0.82,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, format="png", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved pose prediction figure: {args.output}")


def format_vector(vector: torch.Tensor) -> str:
    """Format a three-dimensional tensor for compact figure annotations."""

    return "[" + ", ".join(f"{float(value):.1f}" for value in vector) + "]"


if __name__ == "__main__":
    main()
