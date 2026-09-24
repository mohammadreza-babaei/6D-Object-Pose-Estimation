"""Training and evaluation utilities for Depth-only pose estimation."""

import torch

from src.config import DEVICE
from src.geometry import quaternion_to_rotation_matrix
from src.metrics import add_metric
from src.pose_loss import PoseLoss
from src.depth_pose_model import DepthPoseModel


def create_depth_training_components(
    learning_rate: float = 1e-4,
) -> tuple[DepthPoseModel, PoseLoss, torch.optim.Optimizer]:
    """Create the Depth-only model, loss, and optimizer."""

    model = DepthPoseModel().to(DEVICE)
    criterion = PoseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_depth_one_epoch(
    model: DepthPoseModel,
    dataloader,
    criterion: PoseLoss,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train the Depth-only model for one epoch."""

    model.train()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    total_samples = 0

    for _, depth, target_translation, target_rotation in dataloader:
        depth = depth.to(DEVICE)
        target_translation = target_translation.to(DEVICE)
        target_rotation = target_rotation.to(DEVICE)

        optimizer.zero_grad()
        pred_translation, pred_rotation = model(depth)
        loss, translation_loss, rotation_loss = criterion(
            pred_translation,
            pred_rotation,
            target_translation,
            target_rotation,
        )
        loss.backward()
        optimizer.step()

        batch_size = depth.size(0)
        total_samples += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["translation_loss"] += translation_loss.item() * batch_size
        totals["rotation_loss"] += rotation_loss.item() * batch_size

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return {name: value / total_samples for name, value in totals.items()}


def evaluate_depth(
    model: DepthPoseModel,
    dataloader,
    criterion: PoseLoss,
    model_points: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate Depth-only loss and optionally mean ADD."""

    model.eval()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    total_add = 0.0
    total_samples = 0

    if model_points is not None:
        model_points = model_points.to(DEVICE)

    with torch.no_grad():
        for _, depth, target_translation, target_rotation in dataloader:
            depth = depth.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)
            pred_translation, pred_rotation = model(depth)
            loss, translation_loss, rotation_loss = criterion(
                pred_translation,
                pred_rotation,
                target_translation,
                target_rotation,
            )

            batch_size = depth.size(0)
            total_samples += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["translation_loss"] += translation_loss.item() * batch_size
            totals["rotation_loss"] += rotation_loss.item() * batch_size

            if model_points is not None:
                pred_matrices = quaternion_to_rotation_matrix(pred_rotation)
                target_matrices = quaternion_to_rotation_matrix(target_rotation)
                for index in range(batch_size):
                    total_add += add_metric(
                        model_points,
                        pred_matrices[index],
                        pred_translation[index],
                        target_matrices[index],
                        target_translation[index],
                    ).item()

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    metrics = {name: value / total_samples for name, value in totals.items()}
    if model_points is not None:
        metrics["add"] = total_add / total_samples
    return metrics
