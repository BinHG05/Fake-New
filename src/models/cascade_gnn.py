"""
Multimodal Fake News Detection GNN Model.

Heterogeneous GNN for 6-class classification with binary evaluation support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from typing import Optional, Tuple


class MultiModalFakeNewsGNN(nn.Module):
    """
    Graph Neural Network for Multimodal Fake News Detection.
    
    Architecture:
    1. Input projection layers for text/image features
    2. GNN message passing layers (GAT/SAGE/GCN)
    3. MLP classifier for 6-class output
    
    Supports:
    - 6-class training (TRUE, MOSTLY_TRUE, ..., PANTS_ON_FIRE)
    - Binary evaluation (TRUE vs FAKE)
    """
    
    def __init__(
        self,
        input_dim: int = 1280,  # 768 (text) + 512 (image)
        hidden_dim: int = 256,
        num_classes: int = 6,
        num_layers: int = 2,
        dropout: float = 0.3,
        gnn_type: str = 'gat',  # 'gat', 'sage', or 'gcn'
        heads: int = 4  # For GAT
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (6)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN layer ('gat', 'sage', 'gcn')
            heads: Number of attention heads (for GAT)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            if gnn_type == 'gat':
                # For GAT, output needs to be divided by heads
                self.gnn_layers.append(
                    GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
                )
            elif gnn_type == 'sage':
                self.gnn_layers.append(
                    SAGEConv(in_dim, out_dim)
                )
            else:  # gcn
                self.gnn_layers.append(
                    GCNConv(in_dim, out_dim)
                )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Layer norm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, 1] (optional, for future use)
            
        Returns:
            Logits [N, num_classes]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # GNN layers with residual connections
        for i, gnn in enumerate(self.gnn_layers):
            h_new = gnn(h, edge_index)
            h_new = self.layer_norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Residual connection
            h = h + h_new
        
        # Classifier
        logits = self.classifier(h)
        
        return logits
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions (6-class and binary).
        
        Returns:
            (pred_6class, pred_binary)
        """
        logits = self.forward(x, edge_index, edge_attr)
        pred_6class = logits.argmax(dim=-1)
        
        # Binary: classes 0,1,2 -> 0 (True), classes 3,4,5 -> 1 (Fake)
        pred_binary = (pred_6class >= 3).long()
        
        return pred_6class, pred_binary
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get node embeddings before classifier.
        
        Returns:
            Node embeddings [N, hidden_dim]
        """
        h = self.input_proj(x)
        h = F.relu(h)
        
        for i, gnn in enumerate(self.gnn_layers):
            h_new = gnn(h, edge_index)
            h_new = self.layer_norms[i](h_new)
            h_new = F.relu(h_new)
            h = h + h_new
        
        return h


def create_model(
    input_dim: int = 1280,
    hidden_dim: int = 256,
    num_classes: int = 6,
    gnn_type: str = 'gat'
) -> MultiModalFakeNewsGNN:
    """
    Factory function to create model.
    """
    return MultiModalFakeNewsGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        gnn_type=gnn_type
    )