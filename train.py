"""Train and evaluate the RGB-based 6D pose estimator."""

import json

import torch
from torch.utils.data import DataLoader, random_split

from src.checkpointing import load_checkpoint, save_checkpoint
from src.linemod_dataset import LineMODDataset
from src.model_loader import load_ply_vertices
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.training import (
    create_training_components,
    evaluate,
    train_one_epoch,
)
from src.training_history import plot_training_history, save_training_history


OBJECT_ID = 1
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 42


def main():
    full_train_dataset = LineMODDataset(
        dataset_root=str(DATASET_ROOT),
        object_id=OBJECT_ID,
        split="train",
    )

    validation_size = max(1, int(len(full_train_dataset) * VALIDATION_FRACTION))
    training_size = len(full_train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    test_dataset = LineMODDataset(
        dataset_root=str(DATASET_ROOT),
        object_id=OBJECT_ID,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, criterion, optimizer = create_training_components(LEARNING_RATE)

    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{OBJECT_ID:02d}.ply"
    )

    checkpoint_dir = CHECKPOINT_ROOT
    checkpoint_dir.mkdir(exist_ok=True)

    best_add = float("inf")
    history = []

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"3D model points: {len(model_points)}")
    print("Starting training...")

    for epoch in range(EPOCHS):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            model_points,
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

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_metrics['loss']:.4f}")
        print(f"Validation loss: {validation_metrics['loss']:.4f}")
        print(
            "Translation loss: "
            f"{validation_metrics['translation_loss']:.4f}"
        )
        print(
            "Rotation loss: "
            f"{validation_metrics['rotation_loss']:.4f}"
        )
        print(f"Validation ADD: {validation_metrics['add']:.4f} mm")

        if validation_metrics["add"] < best_add:
            best_add = validation_metrics["add"]

            save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model,
                optimizer,
                epoch=epoch + 1,
                metrics=validation_metrics,
                config={
                    "object_id": OBJECT_ID,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                },
            )

            print("Best model saved.")

    load_checkpoint(checkpoint_dir / "best_model.pth", model)
    test_metrics = evaluate(model, test_loader, criterion, model_points)
    print(f"Final test ADD: {test_metrics['add']:.4f} mm")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_training_history(
        history,
        OUTPUT_ROOT / "rgb_training_history.csv",
    )
    plot_training_history(
        history,
        OUTPUT_ROOT / "figures",
        "rgb_training",
    )
    (OUTPUT_ROOT / "rgb_experiment_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "RGB pose estimation",
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
                    "device": str(next(model.parameters()).device),
                    "num_workers": 0,
                },
                "dataset": {
                    "train_samples": training_size,
                    "validation_samples": validation_size,
                    "test_samples": len(test_dataset),
                    "original_train_samples": len(full_train_dataset),
                    "validation_fraction": VALIDATION_FRACTION,
                    "split": "random_split on the original train split; untouched test split",
                },
                "model": {
                    "backbone": "ResNet-50",
                    "pretrained_weights": "torchvision ResNet50_Weights.DEFAULT (ImageNet)",
                    "input_size": [224, 224],
                    "input_modality": "RGB",
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
                    "checkpoint": str((checkpoint_dir / "best_model.pth").relative_to(OUTPUT_ROOT.parent)),
                },
            },
            indent=2,
        )
    )
    (OUTPUT_ROOT / "rgb_test_metrics.json").write_text(
        json.dumps(
            {
                "object_id": OBJECT_ID,
                "train_samples": training_size,
                "validation_samples": validation_size,
                "test_samples": len(test_dataset),
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()