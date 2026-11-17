"""
Graph-Only Baseline: Simple 2-layer GCN Classifier.

Architecture:
  Input features → Linear projection → GCN × 2 (with residual) → MLP classifier

Simplified version of cascade_gnn.py for baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphOnlyModel(nn.Module):
    """
    Simple 2-layer GCN baseline for graph-level node classification.

    Forward: (x, edge_index) → logits [N, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 128,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, input_dim]  node features
            edge_index: [2, E]          edge list
        Returns:
            logits: [N, num_classes]
        """
        # Project → GCN layers with residual
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h1 = F.relu(self.norm1(self.conv1(h, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h = h + h1  # residual

        h2 = F.relu(self.norm2(self.conv2(h, edge_index)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h = h + h2  # residual

        return self.classifier(h)
