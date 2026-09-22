"""PyTorch dataset utilities for the LineMOD dataset."""

from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.geometry import rotation_matrix_to_quaternion


class LineMODDataset(Dataset):
    """Dataset loader for the LineMOD 6D object pose dataset."""

    def __init__(
        self,
        dataset_root: str,
        object_id: int = 1,
        split: str = "train",
    ):
        self.dataset_root = Path(dataset_root)
        self.object_id = object_id
        self.object_dir = (
            self.dataset_root / "data" / f"{object_id:02d}"
        )

        if split not in {"train", "test"}:
            raise ValueError(
                "split must be either 'train' or 'test'"
            )

        if not self.object_dir.exists():
            raise FileNotFoundError(
                f"Object directory not found: {self.object_dir}"
            )

        split_file = self.object_dir / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        self.sample_ids = [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

        gt_file = self.object_dir / "gt.yml"

        if not gt_file.exists():
            raise FileNotFoundError(
                f"Ground-truth file not found: {gt_file}"
            )

        with open(gt_file, "r") as file:
            self.ground_truth = yaml.safe_load(file)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = int(self.sample_ids[index])

        image_path = (
            self.object_dir
            / "rgb"
            / f"{sample_id:04d}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"RGB image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        annotations = self.ground_truth[sample_id]

        annotation = next(
            (
                item
                for item in annotations
                if int(item["obj_id"]) == self.object_id
            ),
            None,
        )

        if annotation is None:
            raise ValueError(
                f"Object {self.object_id} not found "
                f"in sample {sample_id}"
            )

        # LineMOD bounding box format:
        # [x, y, width, height]
        x, y, box_width, box_height = annotation["obj_bb"]

        image_width, image_height = image.size

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image_width, int(x + box_width))
        y2 = min(image_height, int(y + box_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box for sample {sample_id}: "
                f"{annotation['obj_bb']}"
            )

        # Train the pose estimator on the object crop,
        # matching the YOLO-based inference pipeline.
        image = image.crop((x1, y1, x2, y2))
        image = self.transform(image)

        translation = torch.tensor(
            annotation["cam_t_m2c"],
            dtype=torch.float32,
        )

        rotation_matrix = torch.tensor(
            annotation["cam_R_m2c"],
            dtype=torch.float32,
        ).reshape(3, 3)

        rotation = rotation_matrix_to_quaternion(
            rotation_matrix
        )

        return image, translation, rotation