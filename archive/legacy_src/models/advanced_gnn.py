"""
Advanced Multimodal GNN for Fake News Detection.

Combines frozen text/image encoders with cross-attention fusion
and GraphSAGE message passing for node-level classification.

Architecture:
  XLM-R (frozen) ──→ text_emb (768d)  ┐
                                        ├→ CrossAttentionFusion → (256d)
  ResNet50 (frozen) → img_emb (2048d)  ┘           │
                                                     ↓
                                          GraphSAGE × 2 (with residual)
                                                     ↓
                                           MLP Classifier → [N, 6]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional


class MultimodalGraphNet(nn.Module):
    """
    End-to-end multimodal GNN.

    Takes raw text tokens + image pixels + graph structure → 6-class logits.
    Frozen backbones (only fusion + GNN + classifier are trained).
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 2048,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        fusion_type: str = 'cross_attention',  # 'cross_attention' or 'gated'
        backbone_text: str = 'xlm-roberta-base',
    ):
        super().__init__()

        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.fusion_type = fusion_type

        # --- Frozen Text Backbone ---
        from transformers import AutoModel
        self.text_backbone = AutoModel.from_pretrained(backbone_text)
        for p in self.text_backbone.parameters():
            p.requires_grad = False
        self._text_dim = self.text_backbone.config.hidden_size  # 768

        # --- Frozen Image Backbone ---
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.image_backbone.parameters():
            p.requires_grad = False
        self._image_dim = 2048

        # --- Fusion ---
        from src.models.multimodal_fusion import CrossAttentionFusion, GatedFusion

        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                text_dim=self._text_dim,
                image_dim=self._image_dim,
                fusion_dim=fusion_dim,
            )
        else:
            self.fusion = GatedFusion(
                text_dim=self._text_dim,
                image_dim=self._image_dim,
                fusion_dim=fusion_dim,
            )

        # --- GNN layers (GraphSAGE) ---
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = fusion_dim if i == 0 else hidden_dim
            self.gnn_layers.append(SAGEConv(in_dim, hidden_dim))
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    # ----------------------------------------------------------
    # Feature extraction (frozen)
    # ----------------------------------------------------------

    @torch.no_grad()
    def _extract_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token from frozen XLM-R. → [B, 768]"""
        out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def _extract_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract avgpool features from frozen ResNet. → [B, 2048]"""
        feat = self.image_backbone(images)  # [B, 2048, 1, 1]
        return feat.flatten(1)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        edge_index: torch.Tensor,
        ablation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            input_ids:      [N, seq_len]
            attention_mask: [N, seq_len]
            images:         [N, 3, 224, 224]
            edge_index:     [2, E]
            ablation:       None (full), 'text_only', 'image_only', 'no_graph'

        Returns:
            logits: [N, num_classes]
        """
        # 1. Extract features
        text_emb = self._extract_text(input_ids, attention_mask)   # [N, 768]
        image_emb = self._extract_image(images)                     # [N, 2048]

        # 2. Ablation variants
        if ablation == 'text_only':
            # Zero out image
            image_emb = torch.zeros_like(image_emb)
        elif ablation == 'image_only':
            # Zero out text
            text_emb = torch.zeros_like(text_emb)

        # 3. Fusion
        h = self.fusion(text_emb, image_emb)  # [N, fusion_dim]
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 4. GNN message passing (skip if ablation='no_graph')
        if ablation != 'no_graph' and edge_index.numel() > 0:
            for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
                h_new = gnn(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                # Residual (if dims match)
                if h.shape == h_new.shape:
                    h = h + h_new
                else:
                    h = h_new

        # 5. Classify
        return self.classifier(h)


class PrecomputedMultimodalGNN(nn.Module):
    """
    Variant that works with PRE-COMPUTED embeddings (from existing graph .pt file).

    Use when node features are already stored in data.x as [text_emb || image_emb].
    No frozen backbones — takes concatenated embeddings directly.

    This is compatible with the existing graph at data/04_graph/fakeddit_graph.pt
    where x = [N, 1280] (768 text + 512 image from CLIP).
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        fusion_type: str = 'cross_attention',
    ):
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.dropout = dropout

        # --- Fusion ---
        from src.models.multimodal_fusion import CrossAttentionFusion, GatedFusion

        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                text_dim=text_dim,
                image_dim=image_dim,
                fusion_dim=fusion_dim,
            )
        else:
            self.fusion = GatedFusion(
                text_dim=text_dim,
                image_dim=image_dim,
                fusion_dim=fusion_dim,
            )

        # --- GNN ---
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = fusion_dim if i == 0 else hidden_dim
            self.gnn_layers.append(SAGEConv(in_dim, hidden_dim))
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        ablation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          [N, text_dim + image_dim] precomputed concatenated embeddings
            edge_index: [2, E]
            ablation:   None, 'text_only', 'image_only', 'no_graph'
        Returns:
            logits: [N, num_classes]
        """
        # Split precomputed features
        text_emb = x[:, :self.text_dim]      # [N, 768]
        image_emb = x[:, self.text_dim:]     # [N, 512]

        # Ablation
        if ablation == 'text_only':
            image_emb = torch.zeros_like(image_emb)
        elif ablation == 'image_only':
            text_emb = torch.zeros_like(text_emb)

        # Fusion
        h = self.fusion(text_emb, image_emb)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GNN
        if ablation != 'no_graph' and edge_index.numel() > 0:
            for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
                h_new = gnn(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                if h.shape == h_new.shape:
                    h = h + h_new
                else:
                    h = h_new

        return self.classifier(h)