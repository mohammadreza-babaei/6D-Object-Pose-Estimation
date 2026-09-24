"""Depth-only 6D object pose estimation model."""

import torch
import torch.nn as nn


class DepthPoseModel(nn.Module):
    """Estimate translation and quaternion rotation from a depth crop."""

    def __init__(self):
        super().__init__()

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.translation_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 4)

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.depth_encoder(depth)
        features = torch.flatten(features, 1)
        features = self.fusion(features)
        translation = self.translation_head(features)
        rotation = self.rotation_head(features)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=1)
        return translation, rotation
