"""
Interaction Graph Builder for Multimodal Fake News Detection.

Builds a unified graph with inter-post edges based on Top-K similarity.
"""

import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Label mappings
LABEL_TO_IDX = {
    'TRUE': 0,
    'MOSTLY_TRUE': 1,
    'HALF_TRUE': 2,
    'BARELY_TRUE': 3,
    'FALSE': 4,
    'PANTS_ON_FIRE': 5,
}

# Binary mapping: TRUE/MOSTLY_TRUE/HALF_TRUE -> 0 (True), others -> 1 (Fake)
LABEL_TO_BINARY = {
    'TRUE': 0,
    'MOSTLY_TRUE': 0,
    'HALF_TRUE': 0,
    'BARELY_TRUE': 1,
    'FALSE': 1,
    'PANTS_ON_FIRE': 1,
}


class InteractionGraphBuilder:
    """
    Build interaction graph with Top-K similarity edges.
    
    Features:
    - POST nodes with concatenated [text_emb || image_emb]
    - Top-K text similarity edges
    - Top-K image similarity edges
    - Split masks for train/val/test
    """
    
    def __init__(
        self,
        text_extractor=None,
        image_extractor=None,
        k_text: int = 5,
        k_image: int = 5,
        mode: str = 'prototype'  # 'prototype' or 'scale'
    ):
        """
        Args:
            text_extractor: TextEmbeddingExtractor instance
            image_extractor: ImageEmbeddingExtractor instance
            k_text: Number of text-similar neighbors per node
            k_image: Number of image-similar neighbors per node
            mode: 'prototype' (torch.topk) or 'scale' (FAISS)
        """
        self.text_extractor = text_extractor
        self.image_extractor = image_extractor
        self.k_text = k_text
        self.k_image = k_image
        self.mode = mode
    
    def _compute_topk_edges(
        self, 
        embeddings: torch.Tensor, 
        k: int,
        edge_type: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Top-K similarity edges.
        
        Args:
            embeddings: [N, D] node embeddings
            k: Number of neighbors
            edge_type: Edge type ID (0=text, 1=image)
            
        Returns:
            edge_index: [2, E] tensor
            edge_attr: [E, 1] tensor with edge types
        """
        N = embeddings.size(0)
        
        # Adjust k if larger than available nodes
        k = min(k, N - 1)
        
        if k <= 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.long)
        
        if self.mode == 'prototype' or N < 5000:
            # Full cosine similarity + topk (works for N < 5k)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.mm(embeddings, embeddings.t())
            
            # Set diagonal to -inf to exclude self-loops
            sim_matrix.fill_diagonal_(-float('inf'))
            
            # Get top-k indices for each node
            _, topk_indices = sim_matrix.topk(k, dim=1)
            
            # Build edge_index
            src = torch.arange(N).unsqueeze(1).expand(-1, k).flatten()
            dst = topk_indices.flatten()
            edge_index = torch.stack([src, dst], dim=0)
            
        else:
            # Scale mode: Use FAISS for large N
            try:
                import faiss
                
                embeddings_np = embeddings.numpy().astype('float32')
                faiss.normalize_L2(embeddings_np)
                
                index = faiss.IndexFlatIP(embeddings.size(1))
                index.add(embeddings_np)
                
                # Search k+1 to exclude self
                _, indices = index.search(embeddings_np, k + 1)
                
                # Remove self-loops (first result is usually self)
                topk_indices = torch.from_numpy(indices[:, 1:k+1])
                
                src = torch.arange(N).unsqueeze(1).expand(-1, k).flatten()
                dst = topk_indices.flatten()
                edge_index = torch.stack([src, dst], dim=0)
                
            except ImportError:
                logger.warning("FAISS not installed, falling back to pytorch topk")
                return self._compute_topk_edges(embeddings, k, edge_type)
        
        # Create edge attributes
        num_edges = edge_index.size(1)
        edge_attr = torch.full((num_edges, 1), edge_type, dtype=torch.long)
        
        return edge_index, edge_attr
    
    def build_graph(
        self,
        data_path: str,
        project_root: Optional[str] = None
    ) -> Data:
        """
        Build PyG Data object from merged JSONL.
        
        Args:
            data_path: Path to merged JSONL file
            project_root: Project root for resolving image paths
            
        Returns:
            PyG Data object with:
            - x: [N, text_dim + image_dim] node features
            - edge_index: [2, E] edges
            - edge_attr: [E, 1] edge types
            - y: [N] 6-class labels
            - y_binary: [N] binary labels
            - train_mask, val_mask, test_mask
        """
        data_path = Path(data_path)
        
        if project_root is None:
            # Assume data_path is relative to project root
            project_root = Path(__file__).parent.parent.parent
        else:
            project_root = Path(project_root)
        
        print(f"📖 Loading data from: {data_path}")
        
        # Load records
        records = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        N = len(records)
        print(f"✓ Loaded {N} records")
        
        # Extract texts and image paths
        texts = []
        image_paths = []
        labels = []
        labels_binary = []
        splits = []
        
        for rec in records:
            # Text
            text = rec.get('clean_text', '') or rec.get('raw_text', '') or ''
            texts.append(text)
            
            # Image path
            img_info = rec.get('image_info', {})
            img_path = img_info.get('processed_path', '')
            if img_path:
                # Convert relative path to absolute
                if not Path(img_path).is_absolute():
                    img_path = str(project_root / img_path)
            image_paths.append(img_path)
            
            # Label (6-class)
            label = rec.get('label', 'TRUE').upper()
            labels.append(LABEL_TO_IDX.get(label, 0))
            labels_binary.append(LABEL_TO_BINARY.get(label, 0))
            
            # Split
            splits.append(rec.get('split', 'train'))
        
        # Extract embeddings
        print("🔄 Extracting text embeddings...")
        if self.text_extractor:
            text_embeddings = self.text_extractor.batch_extract(texts)
        else:
            text_embeddings = torch.zeros(N, 768)
        
        print("🔄 Extracting image embeddings...")
        if self.image_extractor:
            image_embeddings = self.image_extractor.batch_extract(image_paths)
        else:
            image_embeddings = torch.zeros(N, 512)
        
        # Concatenate features
        x = torch.cat([text_embeddings, image_embeddings], dim=1)
        print(f"✓ Node features shape: {x.shape}")
        
        # Build edges
        print(f"🔄 Building Top-{self.k_text} text similarity edges...")
        text_edge_index, text_edge_attr = self._compute_topk_edges(
            text_embeddings, self.k_text, edge_type=0
        )
        
        print(f"🔄 Building Top-{self.k_image} image similarity edges...")
        image_edge_index, image_edge_attr = self._compute_topk_edges(
            image_embeddings, self.k_image, edge_type=1
        )
        
        # Combine edges
        edge_index = torch.cat([text_edge_index, image_edge_index], dim=1)
        edge_attr = torch.cat([text_edge_attr, image_edge_attr], dim=0)
        
        print(f"✓ Total edges: {edge_index.size(1)}")
        print(f"  - Text similarity: {text_edge_index.size(1)}")
        print(f"  - Image similarity: {image_edge_index.size(1)}")
        
        # Create masks
        train_mask = torch.tensor([s == 'train' for s in splits])
        val_mask = torch.tensor([s == 'val' for s in splits])
        test_mask = torch.tensor([s == 'test' for s in splits])
        
        print(f"✓ Splits: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(labels, dtype=torch.long),
            y_binary=torch.tensor(labels_binary, dtype=torch.long),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=N
        )
        
        return data
    
    def save_graph(self, graph: Data, output_path: str):
        """Save graph to .pt file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_path)
        print(f"💾 Graph saved to: {output_path}")
    
    @staticmethod
    def load_graph(input_path: str) -> Data:
        """Load graph from .pt file."""
        return torch.load(input_path)