"""Fine-tune a pretrained YOLO detector on the LineMOD dataset."""

from ultralytics import YOLO
import torch

from src.paths import PROJECT_ROOT, YOLO_DATASET_ROOT

DATASET_CONFIG = YOLO_DATASET_ROOT / "dataset.yaml"
MODEL_NAME = PROJECT_ROOT / "yolo11n.pt"

EPOCHS = 20
IMAGE_SIZE = 640
BATCH_SIZE = 8


def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs",
        name="linemod_yolo",
    )


if __name__ == "__main__":
    main()