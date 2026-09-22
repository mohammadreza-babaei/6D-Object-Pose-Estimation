"""PyTorch RGB-D dataset utilities for the LineMOD dataset."""

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.geometry import rotation_matrix_to_quaternion


class LineMODRGBDDataset(Dataset):
    """LineMOD dataset loader returning RGB, depth, and 6D pose."""

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
        info_file = self.object_dir / "info.yml"

        with open(gt_file, "r") as file:
            self.ground_truth = yaml.safe_load(file)

        with open(info_file, "r") as file:
            self.info = yaml.safe_load(file)

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.depth_resize = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = int(self.sample_ids[index])

        rgb_path = (
            self.object_dir
            / "rgb"
            / f"{sample_id:04d}.png"
        )

        depth_path = (
            self.object_dir
            / "depth"
            / f"{sample_id:04d}.png"
        )

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB image not found: {rgb_path}"
            )

        if not depth_path.exists():
            raise FileNotFoundError(
                f"Depth image not found: {depth_path}"
            )

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)

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

        # LineMOD bounding box: [x, y, width, height]
        x, y, box_width, box_height = annotation["obj_bb"]

        image_width, image_height = rgb.size

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image_width, int(x + box_width))
        y2 = min(image_height, int(y + box_height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box for sample {sample_id}: "
                f"{annotation['obj_bb']}"
            )

        # Use exactly the same crop for RGB and depth.
        rgb = rgb.crop((x1, y1, x2, y2))
        depth = depth.crop((x1, y1, x2, y2))

        rgb = self.rgb_transform(rgb)

        depth = self.depth_resize(depth)

        depth = np.array(
            depth,
            dtype=np.float32,
        )

        depth_scale = float(
            self.info[sample_id].get("depth_scale", 1.0)
        )

        # LineMOD translations and model vertices are in millimetres.
        # Keep depth metric, but express it in metres for stable CNN inputs.
        depth = depth * depth_scale / 1000.0

        # Convert depth to a single-channel tensor.
        depth = torch.from_numpy(depth).unsqueeze(0)

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

        return rgb, depth, translation, rotation