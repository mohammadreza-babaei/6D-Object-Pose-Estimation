"""RGB-based 6D object pose estimation model."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class PoseEstimator(nn.Module):
    """Predict 3D translation and quaternion rotation from an RGB image."""

    def __init__(self):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.translation_head = nn.Linear(num_features, 3)
        self.rotation_head = nn.Linear(num_features, 4)

    def forward(self, x):
        features = self.backbone(x)

        translation = self.translation_head(features)
        rotation = self.rotation_head(features)

        # Normalize quaternion to unit length.
        rotation = torch.nn.functional.normalize(rotation, dim=1)

        return translation, rotation