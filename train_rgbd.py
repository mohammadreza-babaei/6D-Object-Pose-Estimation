"""Train, validate, and test the RGB-D pose estimator."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.checkpointing import load_checkpoint, save_checkpoint
from src.config import DEVICE
from src.model_loader import load_ply_vertices
from src.paths import CHECKPOINT_ROOT, DATASET_ROOT, OUTPUT_ROOT
from src.rgbd_dataset import LineMODRGBDDataset
from src.rgbd_training import (
    create_rgbd_training_components,
    evaluate_rgbd,
    train_rgbd_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_ROOT / "rgbd_best.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(args.seed)
    full_train_dataset = LineMODRGBDDataset(
        str(DATASET_ROOT), args.object_id, "train"
    )
    validation_size = max(
        1,
        int(len(full_train_dataset) * args.validation_fraction),
    )
    training_size = len(full_train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [training_size, validation_size],
        generator=generator,
    )
    test_dataset = LineMODRGBDDataset(
        str(DATASET_ROOT), args.object_id, "test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model, criterion, optimizer = create_rgbd_training_components(
        args.learning_rate
    )
    model_points = load_ply_vertices(
        DATASET_ROOT / "models" / f"obj_{args.object_id:02d}.ply"
    )

    best_add = float("inf")
    for epoch in range(args.epochs):
        train_metrics = train_rgbd_one_epoch(
            model, train_loader, criterion, optimizer
        )
        validation_metrics = evaluate_rgbd(
            model, validation_loader, criterion, model_points
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_add={validation_metrics['add']:.3f} mm"
        )

        if validation_metrics["add"] < best_add:
            best_add = validation_metrics["add"]
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch=epoch + 1,
                metrics=validation_metrics,
                config={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            )
            print(f"Saved best checkpoint: {args.checkpoint}")

    load_checkpoint(args.checkpoint, model, device=DEVICE)
    test_metrics = evaluate_rgbd(model, test_loader, criterion, model_points)
    result = {
        "object_id": args.object_id,
        "train_samples": training_size,
        "validation_samples": validation_size,
        "test_samples": len(test_dataset),
        "best_validation_add_mm": best_add,
        "test_metrics": test_metrics,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_ROOT / "rgbd_test_metrics.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved test metrics: {result_path}")


if __name__ == "__main__":
    main()
