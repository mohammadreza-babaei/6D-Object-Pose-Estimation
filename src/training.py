"""Training utilities for 6D pose estimation."""
from src.geometry import quaternion_to_rotation_matrix
from src.metrics import add_metric
import torch

from src.config import DEVICE
from src.pose_loss import PoseLoss
from src.pose_model import PoseEstimator


def create_training_components(learning_rate=1e-4):
    """Create model, loss function, and optimizer."""

    model = PoseEstimator().to(DEVICE)

    criterion = PoseLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    return model, criterion, optimizer
def train_one_epoch(model, dataloader, criterion, optimizer):
    """Train the pose estimator for one epoch and return sample-weighted losses."""

    model.train()
    totals = {"loss": 0.0, "translation_loss": 0.0, "rotation_loss": 0.0}
    total_samples = 0

    for images, target_translation, target_rotation in dataloader:
        images = images.to(DEVICE)
        target_translation = target_translation.to(DEVICE)
        target_rotation = target_rotation.to(DEVICE)

        optimizer.zero_grad()

        pred_translation, pred_rotation = model(images)

        loss, translation_loss, rotation_loss = criterion(
            pred_translation,
            pred_rotation,
            target_translation,
            target_rotation,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["translation_loss"] += translation_loss.item() * batch_size
        totals["rotation_loss"] += rotation_loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return {
        name: value / total_samples
        for name, value in totals.items()
    }
def evaluate(model, dataloader, criterion, model_points=None):
    """Evaluate pose loss and optionally compute mean ADD."""

    model.eval()

    total_loss = 0.0
    total_translation_loss = 0.0
    total_rotation_loss = 0.0
    total_add = 0.0
    total_samples = 0

    if model_points is not None:
        model_points = model_points.to(DEVICE)

    with torch.no_grad():
        for images, target_translation, target_rotation in dataloader:
            images = images.to(DEVICE)
            target_translation = target_translation.to(DEVICE)
            target_rotation = target_rotation.to(DEVICE)

            pred_translation, pred_rotation = model(images)

            loss, translation_loss, rotation_loss = criterion(
                pred_translation,
                pred_rotation,
                target_translation,
                target_rotation,
            )

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_translation_loss += translation_loss.item() * batch_size
            total_rotation_loss += rotation_loss.item() * batch_size
            total_samples += batch_size

            if model_points is not None:
                pred_matrices = quaternion_to_rotation_matrix(
                    pred_rotation
                )

                target_matrices = quaternion_to_rotation_matrix(
                    target_rotation
                )

                for i in range(images.size(0)):
                    add = add_metric(
                        model_points,
                        pred_matrices[i],
                        pred_translation[i],
                        target_matrices[i],
                        target_translation[i],
                    )

                    total_add += add.item()

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    metrics = {
        "loss": total_loss / total_samples,
        "translation_loss": total_translation_loss / total_samples,
        "rotation_loss": total_rotation_loss / total_samples,
    }

    if model_points is not None:
        metrics["add"] = total_add / total_samples

    return metrics