"""
Text-Only Baseline: Frozen XLM-RoBERTa + Linear Classifier.

Architecture:
  xlm-roberta-base (frozen) → [CLS] token (768d) → Linear(768, num_classes)
"""

import torch
import torch.nn as nn


class TextOnlyModel(nn.Module):
    """
    Text-only baseline using frozen XLM-RoBERTa.

    Forward: (input_ids, attention_mask) → logits [B, num_classes]
    """

    def __init__(self, num_classes: int = 6, backbone: str = 'xlm-roberta-base'):
        super().__init__()

        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone)

        # Freeze backbone — chỉ train lớp classifier
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]
        Returns:
            logits: [B, num_classes]
        """
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Lấy [CLS] token (vị trí 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        return self.classifier(cls_emb)
