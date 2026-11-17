"""
CLIP Classifier v2: MLP models for v1 (baseline) and v2 (CLIP+XLM-R).

v1_baseline: 3 hand-crafted text features → 2-class
v2_clip:     CLIP(512) + CLIP_img(512) + XLM-R(768) + mismatch(1) = 1793 → 2-class
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    v1 Baseline: uses only the old hand-crafted text features (3 dims).
    word_count, has_caps_lock, sentiment_score → REAL / FAKE
    """

    def __init__(self, input_dim: int = 3, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLIPMultimodalMLP(nn.Module):
    """
    v2 New: fuses CLIP text, CLIP image, XLM-R text, and mismatch score.

    Input dimensions:
        clip_text_emb:  512
        clip_image_emb: 512
        xlmr_emb:       768
        clip_score:       1
        ─────────────────────
        Total:         1793
    """

    def __init__(
        self,
        clip_dim: int = 512,
        xlmr_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        input_dim = clip_dim + clip_dim + xlmr_dim + 1  # 1793

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),

            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        clip_text: torch.Tensor,
        clip_image: torch.Tensor,
        xlmr: torch.Tensor,
        clip_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            clip_text:  [B, 512]
            clip_image: [B, 512]
            xlmr:       [B, 768]
            clip_score: [B] or [B, 1]
        """
        if clip_score.dim() == 1:
            clip_score = clip_score.unsqueeze(-1)  # [B, 1]

        x = torch.cat([clip_text, clip_image, xlmr, clip_score], dim=-1)  # [B, 1793]
        return self.net(x)


class CLIPOnlyMLP(nn.Module):
    """
    Ablation variant: only CLIP features (no XLM-R).
    clip_text(512) + clip_image(512) + mismatch(1) = 1025 dims
    """

    def __init__(self, clip_dim: int = 512, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        input_dim = clip_dim + clip_dim + 1  # 1025

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        clip_text: torch.Tensor,
        clip_image: torch.Tensor,
        clip_score: torch.Tensor,
    ) -> torch.Tensor:
        if clip_score.dim() == 1:
            clip_score = clip_score.unsqueeze(-1)
        x = torch.cat([clip_text, clip_image, clip_score], dim=-1)
        return self.net(x)


class ExpertFusionMLP(nn.Module):
    """
    v2 Improved (Expert Fusion): Sử dụng cơ chế Gated Attention để tự học trọng số giữa CLIP và XLM-R.
    Điều này giúp model tự 'quyết định' xem với mỗi mẫu tin, nên tin vào nội dung Hình ảnh hay Ngữ cảnh văn bản hơn.
    """
    def __init__(self, clip_dim: int = 512, xlmr_dim: int = 768, hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        # Experts
        self.clip_expert = nn.Sequential(
            nn.Linear(clip_dim * 2 + 1, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        self.xlmr_expert = nn.Sequential(
            nn.Linear(xlmr_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        
        # Gating mechanism (Gated Attention)
        # Input: All features concatenated
        # Output: 2 values summing to 1 (softmax)
        self.gate = nn.Sequential(
            nn.Linear(clip_dim * 2 + xlmr_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, 
        clip_text: torch.Tensor, 
        clip_image: torch.Tensor, 
        xlmr: torch.Tensor, 
        clip_score: torch.Tensor
    ) -> torch.Tensor:
        if clip_score.dim() == 1:
            clip_score = clip_score.unsqueeze(-1)
            
        # CLIP features expert
        clip_feat = torch.cat([clip_text, clip_image, clip_score], dim=-1)
        c_v = self.clip_expert(clip_feat)
        
        # XLM-R features expert
        x_v = self.xlmr_expert(xlmr)
        
        # Get gating weights (Attention weights)
        all_feat = torch.cat([clip_text, clip_image, xlmr, clip_score], dim=-1)
        weights = self.gate(all_feat) # [B, 2]
        
        # Weighted sum (Fusion)
        fused = weights[:, 0:1] * c_v + weights[:, 1:2] * x_v
        
        return self.classifier(fused)
