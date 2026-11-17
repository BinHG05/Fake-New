"""
Cascade Graph Builder.

Constructs propagation graphs (Cascades) from Reddit comment trees.
Each node represents a post/comment.
Edges represent "reply-to" relationships.
Node features are global text embeddings (XLM-R).
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Dict, Optional
import networkx as nx
import logging
from tqdm import tqdm

from src.features.embedding_extractor import TextEmbeddingExtractor

logger = logging.getLogger(__name__)

class CascadeGraphBuilder:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize embedding extractor
        self.text_extractor = TextEmbeddingExtractor(
            model_name='xlm-roberta-base',
            device=self.device
        )
        
    def build_graph(self, item: Dict) -> Optional[Data]:
        """
        Build a PyG Data object from a single news item with cascade.
        
        Args:
            item: Dictionary containing 'raw_text', 'cascade', 'label'
            
        Returns:
            Data object or None if invalid
        """
        post_id = item.get('id')
        root_text = item.get('raw_text', '')
        cascade_nodes = item.get('cascade', [])
        
        # 1. Create NetworkX graph to manage structure
        G = nx.DiGraph()
        
        # Add root node (The Post itself)
        G.add_node("ROOT", text=root_text, user_id=item.get('user_id', 'unknown'), timestamp=item.get('timestamp', 0))
        
        # Add cascade nodes (Comments)
        valid_comments = 0
        
        for comment in cascade_nodes:
            c_id = comment.get('id')
            p_id = comment.get('parent_id')
            text = comment.get('text', '')
            
            if not c_id or not text:
                continue
                
            # If parent is the post ID, link to ROOT
            if p_id == post_id:
                p_id = "ROOT"
            
            G.add_node(c_id, text=text, user_id=comment.get('user_id'), timestamp=comment.get('timestamp'))
            # Edge direction: Comment -> Parent (Information Flow / Reply) or Parent -> Comment?
            # Standard Cascade: Parent -> Comment (Diffusion).
            # But GNN Message Passing often benefits from bi-directional or specific flow.
            # Let's use Source -> Target (Parent -> Child) for diffusion modeling.
            if p_id in G.nodes or p_id == "ROOT":
                 G.add_edge(p_id, c_id)
                 valid_comments += 1
        
        if valid_comments == 0 and len(cascade_nodes) > 0:
            logger.debug(f"Post {post_id}: No valid connected comments found.")
            
        # 2. Convert to PyG Data
        # We need to map node IDs to integers 0..N
        node_ids = list(G.nodes())
        node_mapping = {n: i for i, n in enumerate(node_ids)}
        
        # Extract features for all nodes
        texts = [G.nodes[n].get('text', '') for n in node_ids]
        
        # Batch extract embeddings
        try:
            embeddings = self.text_extractor.batch_extract(texts, batch_size=16)
        except Exception as e:
            logger.error(f"Error extracting embeddings for {post_id}: {e}")
            return None
            
        x = embeddings # [N, 768]
        
        # Create edge_index
        edge_index = []
        for src, dst in G.edges():
            src_idx = node_mapping[src]
            dst_idx = node_mapping[dst]
            edge_index.append([src_idx, dst_idx])
            
            # Make undirected? Usually cascades are directed.
            # But for GCN/GAT, undirected often works better if flow isn't strictly modeled.
            # Let's keep it directed for now, can add reverse edges later or in transform.
            
        if not edge_index:
             # Single node graph (just the root)
             edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
             edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
             
        # Target Label (Graph-level label)
        # Assuming binary or multiclass label is present
        y = None
        if 'label' in item:
            # You might need a mapper here. For now assume it's handled externally or passing raw.
            # Or if it's already an integer.
            pass

        data = Data(x=x, edge_index=edge_index)
        data.post_id = post_id
        data.num_nodes = len(node_ids)
        
        return data

    def process_dataset(self, items: List[Dict]) -> List[Data]:
        """
        Process a list of items into a list of Data objects.
        """
        graph_list = []
        logger.info(f"Building cascade graphs for {len(items)} items...")
        
        for item in tqdm(items, desc="Building Graphs"):
            try:
                data = self.build_graph(item)
                if data:
                    graph_list.append(data)
            except Exception as e:
                logger.error(f"Failed to build graph for item {item.get('id')}: {e}")
                continue
                
        return graph_list