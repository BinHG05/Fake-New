"""
PyTorch Geometric Dataset for Fake News Detection.

Wrapper for loading unified graph with train/val/test masks.
"""

import torch
from torch_geometric.data import Data
from typing import Optional
from pathlib import Path


class FakeNewsGraphDataset:
    """
    Dataset wrapper for unified fake news graph.
    
    Returns single graph with masks for train/val/test splits.
    """
    
    def __init__(
        self,
        graph_path: str = 'data/04_graph/fakeddit_graph.pt',
        device: Optional[str] = None
    ):
        """
        Args:
            graph_path: Path to saved PyG graph (.pt file)
            device: Target device ('cuda', 'cpu', or None for auto)
        """
        self.graph_path = Path(graph_path)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._graph = None
    
    def _load_graph(self):
        """Lazy load graph."""
        if self._graph is not None:
            return
        
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph not found: {self.graph_path}")
        
        print(f"📥 Loading graph from: {self.graph_path}")
        self._graph = torch.load(self.graph_path, weights_only=False)
        print(f"✓ Graph loaded: {self._graph.num_nodes} nodes, {self._graph.num_edges} edges")
    
    @property
    def graph(self) -> Data:
        """Return the full graph."""
        self._load_graph()
        return self._graph
    
    @property
    def train_mask(self) -> torch.Tensor:
        """Return train mask."""
        self._load_graph()
        return self._graph.train_mask
    
    @property
    def val_mask(self) -> torch.Tensor:
        """Return validation mask."""
        self._load_graph()
        return self._graph.val_mask
    
    @property
    def test_mask(self) -> torch.Tensor:
        """Return test mask."""
        self._load_graph()
        return self._graph.test_mask
    
    @property
    def num_classes(self) -> int:
        """Return number of classes (6 for multiclass)."""
        return 6
    
    @property
    def num_features(self) -> int:
        """Return node feature dimension."""
        self._load_graph()
        return self._graph.x.size(1)
    
    def to(self, device: str) -> 'FakeNewsGraphDataset':
        """Move graph to device."""
        self._load_graph()
        self._graph = self._graph.to(device)
        self.device = device
        return self
    
    def get_split_data(self, split: str = 'train') -> dict:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Dict with x, edge_index, edge_attr, y, y_binary, mask
        """
        self._load_graph()
        
        mask_map = {
            'train': self._graph.train_mask,
            'val': self._graph.val_mask,
            'test': self._graph.test_mask
        }
        
        mask = mask_map.get(split, self._graph.train_mask)
        
        return {
            'x': self._graph.x,
            'edge_index': self._graph.edge_index,
            'edge_attr': self._graph.edge_attr,
            'y': self._graph.y,
            'y_binary': self._graph.y_binary,
            'mask': mask
        }
    
    def __repr__(self) -> str:
        self._load_graph()
        return (
            f"FakeNewsGraphDataset(\n"
            f"  path={self.graph_path},\n"
            f"  nodes={self._graph.num_nodes},\n"
            f"  edges={self._graph.num_edges},\n"
            f"  features={self.num_features},\n"
            f"  train={self.train_mask.sum().item()},\n"
            f"  val={self.val_mask.sum().item()},\n"
            f"  test={self.test_mask.sum().item()}\n"
            f")"
        )