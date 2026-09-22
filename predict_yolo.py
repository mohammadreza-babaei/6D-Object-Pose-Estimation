"""Run the trained YOLO detector on LineMOD test images."""

import torch
from ultralytics import YOLO

from src.paths import RUNS_ROOT, YOLO_DATASET_ROOT

MODEL_PATH = RUNS_ROOT / "detect" / "runs" / "linemod_yolo" / "weights" / "best.pt"
SOURCE_DIR = YOLO_DATASET_ROOT / "images" / "val"
OUTPUT_PROJECT = str(RUNS_ROOT / "predict")
OUTPUT_NAME = "linemod_predictions"


def main():
    model = YOLO(MODEL_PATH)

    image_paths = sorted(SOURCE_DIR.glob("*.png"))[:10]

    if not image_paths:
        raise FileNotFoundError(
            f"No validation images found in: {SOURCE_DIR}"
        )

    print(f"Running detection on {len(image_paths)} images...")

    model.predict(
        source=[str(path) for path in image_paths],
        conf=0.25,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        save=True,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
    )

    print("Prediction completed.")
    print(
        f"Results saved to: "
        f"{OUTPUT_PROJECT}/{OUTPUT_NAME}"
    )


if __name__ == "__main__":
    main()