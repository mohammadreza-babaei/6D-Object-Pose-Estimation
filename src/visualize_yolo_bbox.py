"""Visualize a YOLO bounding box on a LineMOD RGB image."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


IMAGE_PATH = Path("data/yolo_linemod/images/train/0004.png")
LABEL_PATH = Path("data/yolo_linemod/labels/train/0004.txt")
OUTPUT_PATH = Path("bbox_visualization.png")


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")
    width, height = image.size

    label = LABEL_PATH.read_text().strip().split()

    class_id = int(label[0])
    x_center = float(label[1]) * width
    y_center = float(label[2]) * height
    box_width = float(label[3]) * width
    box_height = float(label[4]) * height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [(x1, y1), (x2, y2)],
        outline="red",
        width=3,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"YOLO Bounding Box - Class {class_id}")
    plt.axis("off")

    plt.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        dpi=150,
    )

    plt.close()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()