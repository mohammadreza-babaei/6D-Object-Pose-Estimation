"""Train and evaluate the Depth-only 6D pose estimator."""

import json

import torch
from torch.utils.data import DataLoader, random_split

from src.checkpointing import load_checkpoint, save_checkpoint
from src.config import DEVICE
from src.depth_training import (
    create_depth_training_components,
    evaluate_depth,
    train_depth_one_epoch,
)
from src.linemod_dataset import LineMODDataset
from src.model_loader import load_ply_vertices
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.rgbd_dataset import LineMODRGBDDataset
from src.training_history import plot_training_history, save_training_history


OBJECT_ID = 1
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 42
CHECKPOINT_PATH = CHECKPOINT_ROOT / "depth_best.pth"


def main() -> None:
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    full_train_dataset = LineMODRGBDDataset(
        str(DATASET_ROOT), OBJECT_ID, "train"
    )
    validation_size = max(1, int(len(full_train_dataset) * VALIDATION_FRACTION))
    training_size = len(full_train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [training_size, validation_size],
        generator=generator,
    )
    test_dataset = LineMODRGBDDataset(str(DATASET_ROOT), OBJECT_ID, "test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, criterion, optimizer = create_depth_training_components(LEARNING_RATE)
    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{OBJECT_ID:02d}.ply"
    )
    best_add = float("inf")
    history = []

    print(f"Training samples: {training_size}")
    print(f"Validation samples: {validation_size}")
    print(f"Test samples: {len(test_dataset)}")
    print("Starting Depth-only training...")

    for epoch in range(EPOCHS):
        train_metrics = train_depth_one_epoch(
            model, train_loader, criterion, optimizer
        )
        validation_metrics = evaluate_depth(
            model, validation_loader, criterion, model_points
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "train_translation_loss": train_metrics["translation_loss"],
                "train_rotation_loss": train_metrics["rotation_loss"],
                "validation_translation_loss": validation_metrics["translation_loss"],
                "validation_rotation_loss": validation_metrics["rotation_loss"],
                "validation_add": validation_metrics["add"],
            }
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_add={validation_metrics['add']:.3f} mm"
        )

        if validation_metrics["add"] < best_add:
            best_add = validation_metrics["add"]
            save_checkpoint(
                CHECKPOINT_PATH,
                model,
                optimizer,
                epoch=epoch + 1,
                metrics=validation_metrics,
                config={
                    "object_id": OBJECT_ID,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "validation_fraction": VALIDATION_FRACTION,
                    "seed": SPLIT_SEED,
                },
            )

    load_checkpoint(CHECKPOINT_PATH, model, device=DEVICE)
    test_metrics = evaluate_depth(model, test_loader, criterion, model_points)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_training_history(history, OUTPUT_ROOT / "depth_training_history.csv")
    plot_training_history(history, OUTPUT_ROOT / "figures", "depth_training")

    metadata = {
        "experiment": "Depth-only pose estimation",
        "object_id": OBJECT_ID,
        "randomness": {
            "split_seed": SPLIT_SEED,
            "global_torch_seed": None,
            "cuda_seed": None,
            "note": "Only the train/validation split is seeded; model initialization and shuffled DataLoader batches are not explicitly seeded.",
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "device": str(DEVICE),
            "num_workers": 0,
        },
        "dataset": {
            "original_train_samples": len(full_train_dataset),
            "train_samples": training_size,
            "validation_samples": validation_size,
            "test_samples": len(test_dataset),
            "validation_fraction": VALIDATION_FRACTION,
            "split": "random_split on the original train split; untouched test split",
            "crop": "ground-truth object bounding box",
        },
        "model": {
            "architecture": "existing RGB-D depth encoder plus 128->512->256 prediction head",
            "depth_encoder_channels": [1, 32, 64, 128],
            "input_size": [224, 224],
            "input_modality": "depth",
            "pretrained_weights": "none",
            "rotation_representation": "quaternion [w, x, y, z]",
        },
        "loss": {
            "translation": "mean squared error",
            "rotation": "1 - abs(dot(predicted_quaternion, target_quaternion))",
            "translation_weight": criterion.translation_weight,
            "rotation_weight": criterion.rotation_weight,
        },
        "checkpoint_selection": {
            "criterion": "lowest validation ADD",
            "checkpoint": str(CHECKPOINT_PATH),
        },
    }
    (OUTPUT_ROOT / "depth_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "depth_test_metrics.json").write_text(
        json.dumps(
            {
                "object_id": OBJECT_ID,
                "train_samples": training_size,
                "validation_samples": validation_size,
                "test_samples": len(test_dataset),
                "best_validation_add_mm": best_add,
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
