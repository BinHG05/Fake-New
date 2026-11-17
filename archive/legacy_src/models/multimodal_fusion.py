"""
Multimodal Fusion Modules for Fake News Detection.

Implements attention-based fusion strategies that learn to weigh
text vs image signals, solving the "noisy image dominates" problem
observed in Phase 3 simple concat fusion.

Modules:
  - CrossAttentionFusion: Text queries Image via multi-head attention
  - GatedFusion: Learnable σ-gate between modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention: text embedding attends to image embedding.

    Architecture:
      text (768d) → Q projection (fusion_dim)
      image (512d) → K, V projection (fusion_dim)
      attended = MultiHeadAttention(Q=text, K=image, V=image)
      gate = σ(W · [text_proj; attended])
      output = gate * attended + (1-gate) * text_proj

    When image is uninformative, gate → 0 and output ≈ text_proj.
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"

        self.fusion_dim = fusion_dim

        # Project modalities to shared space
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)

        # Multi-head attention (text as Q, image as K/V)
        self.attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gated residual
        self.gate_linear = nn.Linear(fusion_dim * 2, 1)

        # Layer norm
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb:  [B, text_dim]
            image_emb: [B, image_dim]
        Returns:
            fused: [B, fusion_dim]
        """
        # Project to shared dim
        t = self.text_proj(text_emb)    # [B, fusion_dim]
        i = self.image_proj(image_emb)  # [B, fusion_dim]

        # Reshape for attention: [B, 1, fusion_dim] (single-token sequences)
        t_seq = t.unsqueeze(1)  # [B, 1, D]
        i_seq = i.unsqueeze(1)  # [B, 1, D]

        # Cross attention: text queries image
        attended, _ = self.attn(
            query=t_seq, key=i_seq, value=i_seq
        )  # [B, 1, D]
        attended = attended.squeeze(1)  # [B, D]

        # Gated residual: learn to blend text vs attended
        gate = torch.sigmoid(self.gate_linear(torch.cat([t, attended], dim=-1)))  # [B, 1]
        fused = gate * attended + (1 - gate) * t

        return self.norm(fused)


class GatedFusion(nn.Module):
    """
    Simple gated fusion: learnable per-modality weighting.

    Architecture:
      text (text_dim) → proj (fusion_dim)
      image (image_dim) → proj (fusion_dim)
      gate = σ(W · [text_proj; image_proj])
      output = gate * text_proj + (1-gate) * image_proj

    Simpler alternative to CrossAttentionFusion.
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        fusion_dim: int = 256,
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.gate_linear = nn.Linear(fusion_dim * 2, 1)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb:  [B, text_dim]
            image_emb: [B, image_dim]
        Returns:
            fused: [B, fusion_dim]
        """
        t = F.relu(self.text_proj(text_emb))
        i = F.relu(self.image_proj(image_emb))

        gate = torch.sigmoid(self.gate_linear(torch.cat([t, i], dim=-1)))
        fused = gate * t + (1 - gate) * i

        return self.norm(fused)