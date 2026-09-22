"""Detect and crop an object from a LineMOD RGB image using trained YOLO."""

from PIL import Image
from ultralytics import YOLO

from src.paths import OUTPUT_ROOT, RUNS_ROOT, YOLO_DATASET_ROOT

MODEL_PATH = RUNS_ROOT / "detect" / "runs" / "linemod_yolo" / "weights" / "best.pt"
IMAGE_PATH = YOLO_DATASET_ROOT / "images" / "val" / "0000.png"
OUTPUT_DIR = OUTPUT_ROOT / "crops"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        verbose=False,
    )

    image = Image.open(IMAGE_PATH).convert("RGB")

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No object detected.")
        return

    # Use the detection with the highest confidence.
    best_index = boxes.conf.argmax().item()
    box = boxes[best_index]

    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
    confidence = box.conf.item()
    class_id = int(box.cls.item())

    width, height = image.size

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    crop = image.crop((x1, y1, x2, y2))

    output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_crop.png"
    crop.save(output_path)

    print(f"Image: {IMAGE_PATH}")
    print(f"Class ID: {class_id}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
    print(f"Crop size: {crop.size}")
    print(f"Saved crop: {output_path}")


if __name__ == "__main__":
    main()