"""Training and evaluation utilities for RGB-D pose estimation."""

import torch

from src.config import DEVICE
from src.geometry import quaternion_to_rotation_matrix
from src.metrics import add_metric
from src.pose_loss import PoseLoss
from src.rgbd_pose_model import RGBDPoseModel


def create_rgbd_training_components(
    learning_rate: float = 1e-4,
) -> tuple[RGBDPoseModel, PoseLoss, torch.optim.Optimizer]:
    """Create the RGB-D model, loss, and optimizer on the selected device."""

    model = RGBDPoseModel().to(DEVICE)
    criterion = PoseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_rgbd_one_epoch(
    model: RGBDPoseModel,
    dataloader,
    criterion: PoseLoss,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train the RGB-D model for one epoch and return sample-weighted losses."""

    model.train()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    total_samples = 0

    for rgb, depth, target_translation, target_rotation in dataloader:
        rgb = rgb.to(DEVICE)
        depth = depth.to(DEVICE)
        target_translation = target_translation.to(DEVICE)
        target_rotation = target_rotation.to(DEVICE)

        optimizer.zero_grad()
        pred_translation, pred_rotation = model(rgb, depth)
        loss, translation_loss, rotation_loss = criterion(
            pred_translation,
            pred_rotation,
            target_translation,
            target_rotation,
        )
        loss.backward()
        optimizer.step()

        batch_size = rgb.size(0)
        total_samples += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["translation_loss"] += translation_loss.item() * batch_size
        totals["rotation_loss"] += rotation_loss.item() * batch_size

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return {name: value / total_samples for name, value in totals.items()}


def evaluate_rgbd(
    model: RGBDPoseModel,
    dataloader,
    criterion: PoseLoss,
    model_points: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate RGB-D loss and optional mean ADD in millimetres."""

    model.eval()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    total_add = 0.0
    total_samples = 0

    if model_points is not None:
        model_points = model_points.to(DEVICE)

    with torch.no_grad():
        for rgb, depth, target_translation, target_rotation in dataloader:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)

            pred_translation, pred_rotation = model(rgb, depth)
            loss, translation_loss, rotation_loss = criterion(
                pred_translation,
                pred_rotation,
                target_translation,
                target_rotation,
            )

            batch_size = rgb.size(0)
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

    metrics = {
        name: value / total_samples
        for name, value in totals.items()
    }
    if model_points is not None:
        metrics["add"] = total_add / total_samples
    return metrics
