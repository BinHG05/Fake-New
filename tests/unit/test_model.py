"""
Unit tests for MultimodalCascadeGNN architecture.
"""

import torch
from src.fakenews.models.multimodal_gnn import MultimodalCascadeGNN, CrossAttentionFusion


def test_cross_attention_fusion_shape():
    batch_size = 4
    fusion = CrossAttentionFusion(text_dim=768, image_dim=1025, fusion_dim=512)
    text = torch.randn(batch_size, 768)
    image = torch.randn(batch_size, 1025)
    fused = fusion(text, image)
    assert fused.shape == (batch_size, 512)


def test_multimodal_cascade_gnn_forward():
    num_nodes = 5
    x = torch.randn(num_nodes, 1793)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    model = MultimodalCascadeGNN(
        input_dim=1793,
        text_dim=768,
        image_dim=1025,
        fusion_dim=256,
        hidden_dim=256,
        num_classes=2,
        num_gnn_layers=2,
        dropout=0.0,
    )
    model.eval()
    logits = model(x, edge_index, batch)
    assert logits.shape == (1, 2)
