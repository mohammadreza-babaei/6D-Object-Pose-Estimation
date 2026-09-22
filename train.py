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


OBJECT_ID = 1
BATCH_SIZE = 4
EPOCHS = 20
VALIDATION_FRACTION = 0.2


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
        generator=torch.Generator().manual_seed(42),
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

    model, criterion, optimizer = create_training_components()

    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{OBJECT_ID:02d}.ply"
    )

    checkpoint_dir = CHECKPOINT_ROOT
    checkpoint_dir.mkdir(exist_ok=True)

    best_add = float("inf")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"3D model points: {len(model_points)}")
    print("Starting training...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
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

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
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