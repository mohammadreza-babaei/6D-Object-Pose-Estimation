"""Prepare the LineMOD dataset for YOLO object detection."""

import shutil

import yaml
from PIL import Image

from src.paths import DATASET_ROOT, YOLO_DATASET_ROOT

OUTPUT_ROOT = YOLO_DATASET_ROOT

OBJECT_ID = 1


def convert_bbox_to_yolo(bbox, image_width, image_height):
    """Convert [x, y, width, height] to normalized YOLO format."""

    x, y, width, height = bbox

    x_center = (x + width / 2.0) / image_width
    y_center = (y + height / 2.0) / image_height

    normalized_width = width / image_width
    normalized_height = height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )


def prepare_split(object_dir, ground_truth, split):
    """Prepare images and labels for one YOLO split."""

    split_file = object_dir / f"{split}.txt"
    sample_ids = split_file.read_text().splitlines()

    yolo_split = "train" if split == "train" else "val"

    images_dir = OUTPUT_ROOT / "images" / yolo_split
    labels_dir = OUTPUT_ROOT / "labels" / yolo_split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in sample_ids:
        sample_id = int(sample)

        source_image = (
            object_dir
            / "rgb"
            / f"{sample_id:04d}.png"
        )

        destination_image = (
            images_dir
            / f"{sample_id:04d}.png"
        )

        with Image.open(source_image) as image:
            image_width, image_height = image.size

        annotation = next(
            item
            for item in ground_truth[sample_id]
            if int(item["obj_id"]) == OBJECT_ID
        )

        bbox = annotation["obj_bb"]

        (
            x_center,
            y_center,
            bbox_width,
            bbox_height,
        ) = convert_bbox_to_yolo(
            bbox,
            image_width,
            image_height,
        )

        # We currently train YOLO for one object class.
        class_id = 0

        label = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{bbox_width:.6f} "
            f"{bbox_height:.6f}\n"
        )

        label_path = (
            labels_dir
            / f"{sample_id:04d}.txt"
        )

        label_path.write_text(label)

        shutil.copy2(
            source_image,
            destination_image,
        )

    print(
        f"{yolo_split}: "
        f"{len(sample_ids)} samples prepared."
    )


def main():
    object_dir = (
        DATASET_ROOT
        / "data"
        / f"{OBJECT_ID:02d}"
    )

    gt_path = object_dir / "gt.yml"

    with open(gt_path, "r") as file:
        ground_truth = yaml.safe_load(file)

    prepare_split(
        object_dir,
        ground_truth,
        "train",
    )

    prepare_split(
        object_dir,
        ground_truth,
        "test",
    )

    print("\nYOLO dataset created successfully.")
    print(f"Location: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()