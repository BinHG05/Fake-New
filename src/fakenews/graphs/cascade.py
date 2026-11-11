"""
Cascade Graph Builder for propagation trees.
Constructs PyG Data objects with 1793d multimodal features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import torch
from torch_geometric.data import Data

from src.fakenews.features.embeddings import CLIPFeatureExtractor, TextEmbeddingExtractor

logger = logging.getLogger(__name__)


class CascadeGraphBuilder:
    """
    Constructs cascade propagation graphs.
    Node 0: Root post (768d text embedding + 1025d CLIP features = 1793d)
    Node 1..k: Comments (768d text embedding + 1025d zero padding = 1793d)
    """

    def __init__(
        self,
        text_extractor: Optional[TextEmbeddingExtractor] = None,
        clip_extractor: Optional[CLIPFeatureExtractor] = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_extractor = text_extractor or TextEmbeddingExtractor(device=self.device)
        self.clip_extractor = clip_extractor or CLIPFeatureExtractor(device=self.device)

    def build_graph(
        self,
        item: dict[str, Any],
        image_path: str | Path | None = None,
    ) -> Data:
        post_id = str(item.get("id", "sample")).strip()
        root_text = item.get("clean_text") or item.get("raw_text") or item.get("title") or ""
        cascade_nodes = item.get("cascade") or []

        # Resolve image path if not passed explicitly
        if image_path is None:
            media = item.get("media")
            if isinstance(media, dict) and media.get("processed_path"):
                image_path = media["processed_path"]
            elif item.get("image"):
                image_path = item["image"]

        # 1. Root Post Multimodal Node (1793d)
        root_text_emb = self.text_extractor.extract(root_text)  # [768]
        root_clip_emb = self.clip_extractor.extract(root_text, image_path)  # [1025]
        root_feature = torch.cat([root_text_emb, root_clip_emb], dim=-1).unsqueeze(0)  # [1, 1793]

        node_features = [root_feature]
        node_id_to_idx = {post_id: 0, "ROOT": 0}
        edges: list[list[int]] = []

        # 2. Add Cascade Comments
        for comment in cascade_nodes:
            c_id = str(comment.get("id", "")).strip()
            text = str(comment.get("text", "")).strip()
            if not c_id or not text:
                continue

            c_idx = len(node_features)
            node_id_to_idx[c_id] = c_idx

            c_text_emb = self.text_extractor.extract(text)  # [768]
            c_clip_zero = torch.zeros(1025)  # Comments have no independent image
            c_feature = torch.cat([c_text_emb, c_clip_zero], dim=-1).unsqueeze(0)
            node_features.append(c_feature)

            parent_id = str(comment.get("parent_id", "")).strip()
            # If parent is post or unknown, attach to root (0)
            p_idx = node_id_to_idx.get(parent_id, 0)
            edges.append([p_idx, c_idx])

        # 3. Assemble PyG Data
        x = torch.cat(node_features, dim=0).float()
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, post_id=post_id, num_nodes=int(x.size(0)))
        return data

