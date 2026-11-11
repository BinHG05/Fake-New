"""
Canonical Multimodal Cascade GNN Model.

Architecture:
  Text (768d) + Image (1025d)
        │
  CrossAttention / Gated Fusion -> (fusion_dim)
        │
  GATv2 Message Passing with Residual Skip Connections
        │
  Hybrid Pooling (Global Mean Pool + Global Max Pool) + Global Root Skip
        │
  Classifier MLP -> Logits (REAL: 0, FAKE: 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class CrossAttentionFusion(nn.Module):
    """Text queries Image via multi-head attention with gated residual connection."""

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 1025,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_dim % num_heads != 0:
            raise ValueError(f"fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads})")

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate_linear = nn.Linear(fusion_dim * 2, 1)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb)
        attended, _ = self.attn(t.unsqueeze(1), i.unsqueeze(1), i.unsqueeze(1))
        attended = attended.squeeze(1)
        gate = torch.sigmoid(self.gate_linear(torch.cat([t, attended], dim=-1)))
        return self.norm(gate * attended + (1.0 - gate) * t)


class GatedFusion(nn.Module):
    """Simple gated fusion between text and image modalities."""

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 1025,
        fusion_dim: int = 512,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.gate_linear = nn.Linear(fusion_dim * 2, 1)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        t = F.relu(self.text_proj(text_emb))
        i = F.relu(self.image_proj(image_emb))
        gate = torch.sigmoid(self.gate_linear(torch.cat([t, i], dim=-1)))
        return self.norm(gate * t + (1.0 - gate) * i)


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing for class-imbalanced fake news classification."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        smoothing: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            if self.smoothing > 0:
                true_dist.fill_(self.smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
            else:
                true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0)

        log_pt = F.log_softmax(inputs, dim=-1)
        ce_loss = torch.sum(-true_dist * log_pt, dim=-1)

        pt = torch.exp(F.log_softmax(inputs, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(-1))

        if self.alpha is not None:
            at = self.alpha.gather(0, targets.data)
            focal_loss = at * ((1.0 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


class MultimodalCascadeGNN(nn.Module):
    """
    Canonical Multimodal Cascade GNN.
    Combines text and image modalities with cascade propagation structure.
    """

    def __init__(
        self,
        input_dim: int = 1793,
        text_dim: int = 768,
        image_dim: int = 1025,
        fusion_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 2,
        num_gnn_layers: int = 3,
        dropout: float = 0.3,
        gnn_type: str = "gat",
        fusion_type: str = "cross_attention",
        learn_missing_image_emb: bool = False,
    ) -> None:
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.has_image = image_dim > 0
        self.dropout_rate = dropout
        self.gnn_type = gnn_type
        self.learn_missing_image_emb = bool(learn_missing_image_emb)

        if self.has_image:
            if self.learn_missing_image_emb:
                self.missing_image_emb = nn.Parameter(torch.randn(image_dim) * 0.02)
            if fusion_type == "cross_attention":
                self.fusion = CrossAttentionFusion(
                    text_dim=text_dim,
                    image_dim=image_dim,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                )
            else:
                self.fusion = GatedFusion(
                    text_dim=text_dim,
                    image_dim=image_dim,
                    fusion_dim=fusion_dim,
                )
            self.fusion_type = fusion_type
            gnn_input_dim = fusion_dim
        else:
            self.input_proj = nn.Linear(input_dim, fusion_dim)
            gnn_input_dim = fusion_dim

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        effective_layers = max(1, int(num_gnn_layers))
        for i in range(effective_layers):
            in_dim = gnn_input_dim if i == 0 else hidden_dim
            self.convs.append(
                GATv2Conv(in_dim, hidden_dim // 8, heads=8, concat=True, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.residual_proj = (
            nn.Linear(gnn_input_dim, hidden_dim)
            if gnn_input_dim != hidden_dim
            else nn.Identity()
        )

        # Classification head: (hidden_dim * 2 from Mean + Max pool) + (gnn_input_dim from skip)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + gnn_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        ablation: str | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if self.has_image:
            text_emb = x[:, : self.text_dim]
            image_emb = x[:, self.text_dim :]

            if ablation == "text_only":
                image_emb = torch.zeros_like(image_emb)
            elif ablation == "image_only":
                text_emb = torch.zeros_like(text_emb)
            elif ablation == "missing_image":
                image_emb = torch.zeros_like(image_emb)

            if self.learn_missing_image_emb:
                missing_mask = image_emb.abs().sum(dim=-1, keepdim=True) == 0
                if missing_mask.any():
                    image_emb = torch.where(
                        missing_mask,
                        self.missing_image_emb.unsqueeze(0).expand_as(image_emb),
                        image_emb,
                    )
            h = self.fusion(text_emb, image_emb)
        else:
            h = F.relu(self.input_proj(x))

        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GNN Forward with residual connections
        curr_h = h
        layer_attentions = []
        if ablation != "no_graph":
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                if return_attention:
                    h_new, (ei, alpha) = conv(curr_h, edge_index, return_attention_weights=True)
                    layer_attentions.append({"edge_index": ei, "alpha": alpha})
                else:
                    h_new = conv(curr_h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)

                if i == 0:
                    curr_h = h_new + self.residual_proj(curr_h)
                else:
                    curr_h = h_new + curr_h

        # Hybrid pooling (Mean + Max)
        mean_p = global_mean_pool(curr_h, batch)
        max_p = global_max_pool(curr_h, batch)
        graph_emb = torch.cat([mean_p, max_p], dim=-1)

        # Skip connection from pre-GNN fused embedding
        skip_emb = global_mean_pool(h, batch)

        final_emb = torch.cat([graph_emb, skip_emb], dim=-1)
        logits = self.classifier(final_emb)

        if return_attention:
            return logits, {
                "layer_attentions": layer_attentions,
                "node_embeddings": curr_h.detach(),
            }
        return logits

