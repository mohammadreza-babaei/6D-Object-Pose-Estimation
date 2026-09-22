"""Checkpoint helpers shared by pose experiments."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save model state and enough metadata to resume or reproduce an experiment."""

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a new-format checkpoint, while accepting legacy state-dict files."""

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    model.load_state_dict(checkpoint)
    return {
        "version": 0,
        "epoch": None,
        "metrics": {},
        "config": {},
    }
