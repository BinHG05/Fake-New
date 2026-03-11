"""
Image-Only Baseline: Frozen ResNet-50 + Linear Classifier.

Architecture:
  resnet50 (frozen, avgpool output 2048d) → Linear(2048, num_classes)
"""

import torch
import torch.nn as nn


class ImageOnlyModel(nn.Module):
    """
    Image-only baseline using frozen ResNet-50.

    Forward: images [B, 3, 224, 224] → logits [B, num_classes]
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        from torchvision import models

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Lấy tất cả layers trừ fc cuối
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 2048, 1, 1]

        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            logits: [B, num_classes]
        """
        with torch.no_grad():
            features = self.features(images)  # [B, 2048, 1, 1]

        features = features.flatten(1)  # [B, 2048]
        return self.classifier(features)
