"""
Baseline models for fake news detection comparison.
Includes:
- CLIP-only classifier (Feature baseline)
- Text-only classifier
- Early-fusion classifier
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextOnlyBaseline(nn.Module):
    """Text-only classifier using 768d text embeddings."""

    def __init__(self, text_dim: int = 768, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CLIPOnlyClassifier(nn.Module):
    """
    CLIP-only classifier operating on 1025d features:
    [CLIP Text (512) | CLIP Image (512) | CLIP Similarity (1)]
    """

    def __init__(self, input_dim: int = 1025, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EarlyFusionBaseline(nn.Module):
    """Early fusion baseline concatenating text (768d) and image (1025d)."""

    def __init__(self, text_dim: int = 768, image_dim: int = 1025, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

