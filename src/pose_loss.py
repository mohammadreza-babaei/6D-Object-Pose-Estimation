"""Loss functions for 6D object pose estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    """Combined translation and quaternion rotation loss."""

    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight

    def forward(
        self,
        pred_translation,
        pred_rotation,
        target_translation,
        target_rotation,
    ):
        # Translation loss
        translation_loss = F.mse_loss(
            pred_translation,
            target_translation,
        )

        # Normalize quaternions
        pred_rotation = F.normalize(pred_rotation, dim=1)
        target_rotation = F.normalize(target_rotation, dim=1)

        # q and -q represent the same 3D rotation.
        similarity = torch.abs(
            torch.sum(pred_rotation * target_rotation, dim=1)
        )
        rotation_loss = (1.0 - similarity).mean()

        total_loss = (
            self.translation_weight * translation_loss
            + self.rotation_weight * rotation_loss
        )

        return total_loss, translation_loss, rotation_loss