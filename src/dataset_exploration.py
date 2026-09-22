"""Utilities for exploring the LineMOD dataset."""

from pathlib import Path

import yaml


def inspect_dataset(dataset_path: str) -> None:
    """Print LineMOD modalities, split sizes, and annotation metadata."""

    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    print(f"Dataset root: {root}")
    print("\nContents:")

    for item in sorted(root.iterdir()):
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"[{item_type}] {item.name}")

    data_root = root / "data"
    models_root = root / "models"
    if not data_root.exists():
        return

    print("\nObjects:")
    for object_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        train_file = object_dir / "train.txt"
        test_file = object_dir / "test.txt"
        gt_file = object_dir / "gt.yml"
        info_file = object_dir / "info.yml"

        train_count = len(train_file.read_text().splitlines()) if train_file.exists() else 0
        test_count = len(test_file.read_text().splitlines()) if test_file.exists() else 0
        print(
            f"{object_dir.name}: train={train_count}, test={test_count}, "
            f"rgb={len(list((object_dir / 'rgb').glob('*.png')))}, "
            f"depth={len(list((object_dir / 'depth').glob('*.png')))}, "
            f"mask={len(list((object_dir / 'mask').glob('*.png')))}"
        )

        if gt_file.exists() and info_file.exists():
            ground_truth = yaml.safe_load(gt_file.read_text())
            info = yaml.safe_load(info_file.read_text())
            first_id = sorted(ground_truth)[0]
            first_annotation = ground_truth[first_id][0]
            print(
                f"  sample {first_id}: bbox={first_annotation['obj_bb']}, "
                f"translation_mm={first_annotation['cam_t_m2c']}"
            )
            print(f"  camera_K={info[first_id]['cam_K']}")

    if models_root.exists():
        print(f"\n3D models: {len(list(models_root.glob('obj_*.ply')))} PLY files")


if __name__ == "__main__":
    DATASET_PATH = "data/linemod/Linemod_preprocessed"
    inspect_dataset(DATASET_PATH)