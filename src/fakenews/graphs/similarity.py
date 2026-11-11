"""
Similarity Graph Builder.
Constructs cosine similarity graphs between posts for transductive baselines.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def build_similarity_graph(features: torch.Tensor, k: int = 5, threshold: float = 0.7) -> Data:
    """Builds a k-nearest neighbor similarity graph from feature matrix."""
    normed = torch.nn.functional.normalize(features, p=2, dim=-1)
    sim_matrix = torch.mm(normed, normed.t())

    num_nodes = features.size(0)
    edges: list[list[int]] = []

    for i in range(num_nodes):
        sims, indices = torch.topk(sim_matrix[i], k=min(k + 1, num_nodes))
        for sim, idx in zip(sims, indices):
            idx_int = int(idx)
            if idx_int != i and sim.item() >= threshold:
                edges.append([i, idx_int])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    return Data(x=features, edge_index=edge_index, num_nodes=num_nodes)

