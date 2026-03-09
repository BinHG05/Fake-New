"""
Simple Fusion Baseline: Frozen BERT + Frozen ResNet → Concat → MLP.

Architecture:
  xlm-roberta-base (frozen) → [CLS] (768d)  ┐
                                              ├→ concat (2816d) → MLP → num_classes
  resnet50 (frozen)         → avgpool (2048d) ┘
"""

import torch
import torch.nn as nn


class SimpleFusionModel(nn.Module):
    """
    Early-fusion baseline: concatenate frozen text + image embeddings → MLP.

    Forward: (input_ids, attention_mask, images) → logits [B, num_classes]
    """

    def __init__(self, num_classes: int = 6, backbone_text: str = 'xlm-roberta-base'):
        super().__init__()

        # --- Text backbone (frozen) ---
        from transformers import AutoModel
        self.text_backbone = AutoModel.from_pretrained(backbone_text)
        for param in self.text_backbone.parameters():
            param.requires_grad = False
        text_dim = self.text_backbone.config.hidden_size  # 768

        # --- Image backbone (frozen) ---
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])  # → [B,2048,1,1]
        for param in self.image_backbone.parameters():
            param.requires_grad = False
        image_dim = 2048

        # --- Classifier (trainable) ---
        fused_dim = text_dim + image_dim  # 2816
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]
            images:         [B, 3, 224, 224]
        Returns:
            logits: [B, num_classes]
        """
        # Text embedding
        with torch.no_grad():
            text_out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_out.last_hidden_state[:, 0, :]  # [B, 768]

        # Image embedding
        with torch.no_grad():
            img_feat = self.image_backbone(images)  # [B, 2048, 1, 1]
        img_emb = img_feat.flatten(1)  # [B, 2048]

        # Concat + classify
        fused = torch.cat([text_emb, img_emb], dim=1)  # [B, 2816]
        return self.classifier(fused)
