"""Evaluation metrics for 6D object pose estimation."""

import torch


def add_metric(
    model_points,
    pred_rotation,
    pred_translation,
    target_rotation,
    target_translation,
):
    """Compute ADD between predicted and ground-truth poses."""

    pred_points = (
        model_points @ pred_rotation.T
        + pred_translation
    )

    target_points = (
        model_points @ target_rotation.T
        + target_translation
    )

    distances = torch.norm(
        pred_points - target_points,
        dim=1,
    )

    return distances.mean()