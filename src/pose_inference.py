"""Run 6D pose inference on an object crop detected by YOLO."""

import torch
from PIL import Image
from torchvision import transforms

from src.checkpointing import load_checkpoint
from src.config import DEVICE
from src.paths import CHECKPOINT_ROOT, OUTPUT_ROOT
from src.pose_model import PoseEstimator


CHECKPOINT_PATH = CHECKPOINT_ROOT / "best_model.pth"
CROP_PATH = OUTPUT_ROOT / "crops" / "0000_crop.png"


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not CROP_PATH.exists():
        raise FileNotFoundError(
            f"Crop image not found: {CROP_PATH}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(CROP_PATH).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = PoseEstimator().to(DEVICE)

    load_checkpoint(CHECKPOINT_PATH, model, device=DEVICE)
    model.eval()

    with torch.no_grad():
        translation, quaternion = model(image_tensor)

    translation = translation.squeeze(0).cpu()
    quaternion = quaternion.squeeze(0).cpu()

    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion
    ).clamp_min(1e-8)

    print(f"Input crop: {CROP_PATH}")
    print(f"Device: {DEVICE}")

    print("\nPredicted translation (mm):")
    print(f"x = {translation[0].item():.4f}")
    print(f"y = {translation[1].item():.4f}")
    print(f"z = {translation[2].item():.4f}")

    print("\nPredicted rotation quaternion:")
    print(f"qw = {quaternion[0].item():.6f}")
    print(f"qx = {quaternion[1].item():.6f}")
    print(f"qy = {quaternion[2].item():.6f}")
    print(f"qz = {quaternion[3].item():.6f}")

    print(
        "\nQuaternion norm: "
        f"{torch.linalg.vector_norm(quaternion).item():.6f}"
    )


if __name__ == "__main__":
    main()