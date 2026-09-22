"""RGB-D fusion model for 6D object pose estimation."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class RGBDPoseModel(nn.Module):
    """Estimate translation and rotation from RGB and depth images."""

    def __init__(self):
        super().__init__()

        # RGB branch: pretrained ResNet-50
        rgb_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.rgb_encoder = nn.Sequential(
            *list(rgb_backbone.children())[:-1]
        )

        # Depth branch: lightweight CNN
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

        # RGB features: 2048
        # Depth features: 128
        fused_features = 2048 + 128

        self.fusion = nn.Sequential(
            nn.Linear(fused_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # Translation: x, y, z
        self.translation_head = nn.Linear(256, 3)

        # Rotation represented as quaternion: qw, qx, qy, qz
        self.rotation_head = nn.Linear(256, 4)

    def forward(self, rgb, depth):
        rgb_features = self.rgb_encoder(rgb)
        rgb_features = torch.flatten(rgb_features, 1)

        depth_features = self.depth_encoder(depth)
        depth_features = torch.flatten(depth_features, 1)

        features = torch.cat(
            [rgb_features, depth_features],
            dim=1,
        )

        features = self.fusion(features)

        translation = self.translation_head(features)

        quaternion = self.rotation_head(features)
        quaternion = torch.nn.functional.normalize(
            quaternion,
            p=2,
            dim=1,
        )

        return translation, quaternion