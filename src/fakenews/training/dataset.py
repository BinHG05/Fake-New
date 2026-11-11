"""
Dataset loader and graph augmentation for multimodal cascade training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.fakenews.labeling.labels import BINARY_LABEL_MAP, derive_binary_label

logger = logging.getLogger(__name__)


def augment_graph(
    graph: Data,
    node_drop_prob: float = 0.0,
    edge_drop_prob: float = 0.0,
    feature_mask_prob: float = 0.0,
    seed: int | None = None,
) -> Data:
    """Applies dropout augmentation to comment nodes and edges while preserving root post."""
    num_nodes = int(graph.num_nodes)
    if num_nodes <= 1:
        return graph

    rng = np.random.default_rng(seed)
    keep_nodes = np.arange(num_nodes)

    if node_drop_prob > 0:
        comment_nodes = np.arange(1, num_nodes)
        keep_comment_mask = rng.random(comment_nodes.shape[0]) >= node_drop_prob
        keep_nodes = np.concatenate(([0], comment_nodes[keep_comment_mask]))
        keep_nodes = np.unique(keep_nodes)

    node_map = {int(old): new for new, old in enumerate(keep_nodes.tolist())}
    new_x = graph.x[keep_nodes].clone()

    new_edge_index = graph.edge_index
    if edge_drop_prob > 0 or len(keep_nodes) != num_nodes:
        edges = graph.edge_index.detach().cpu().numpy().T
        filtered_edges = []
        for src, dst in edges:
            src_i, dst_i = int(src), int(dst)
            if src_i in node_map and dst_i in node_map:
                if edge_drop_prob <= 0 or rng.random() >= edge_drop_prob:
                    filtered_edges.append([node_map[src_i], node_map[dst_i]])

        if filtered_edges:
            new_edge_index = torch.tensor(filtered_edges, dtype=torch.long).T.contiguous()
        else:
            new_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    if feature_mask_prob > 0:
        mask = torch.rand_like(new_x) < feature_mask_prob
        new_x = new_x.masked_fill(mask, 0.0)

    new_data = Data(x=new_x, edge_index=new_edge_index, y=graph.y)
    new_data.num_nodes = int(new_x.size(0))
    for key in graph.keys():
        if key not in ("x", "edge_index", "y", "num_nodes"):
            try:
                setattr(new_data, key, getattr(graph, key))
            except Exception:
                pass
    return new_data


class AugmentedGraphDataset:
    """Dataset wrapper providing on-the-fly graph augmentation."""

    def __init__(
        self,
        graphs: list[Data],
        node_drop_prob: float = 0.0,
        edge_drop_prob: float = 0.0,
        feature_mask_prob: float = 0.0,
        base_seed: int = 42,
    ) -> None:
        self.graphs = graphs
        self.node_drop_prob = node_drop_prob
        self.edge_drop_prob = edge_drop_prob
        self.feature_mask_prob = feature_mask_prob
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        graph = self.graphs[idx]
        if self.node_drop_prob <= 0 and self.edge_drop_prob <= 0 and self.feature_mask_prob <= 0:
            return graph
        return augment_graph(
            graph,
            node_drop_prob=self.node_drop_prob,
            edge_drop_prob=self.edge_drop_prob,
            feature_mask_prob=self.feature_mask_prob,
            seed=self.base_seed + idx,
        )


def load_cascade_graphs(
    graph_dir: str | Path,
    metadata_path: str | Path,
    limit: int | None = None,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Loads cascade graph files and pairs them with metadata split and binary labels."""
    gdir = Path(graph_dir)
    mpath = Path(metadata_path)

    id_to_meta: dict[str, dict[str, Any]] = {}
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            try:
                label_bin, label_idx = derive_binary_label(item)
            except ValueError:
                continue
            id_to_meta[str(item["id"])] = {
                "label": label_idx,
                "split": str(item.get("split", "train")).lower(),
            }

    torch.serialization.add_safe_globals([Data])
    train_graphs: list[Data] = []
    val_graphs: list[Data] = []
    test_graphs: list[Data] = []

    pt_files = sorted(gdir.glob("*.pt"))
    if limit is not None and limit > 0:
        pt_files = pt_files[:limit]

    for pt_file in pt_files:
        post_id = pt_file.stem
        if post_id not in id_to_meta:
            continue
        try:
            data = torch.load(pt_file, weights_only=False)
            meta = id_to_meta[post_id]
            data.y = torch.tensor([meta["label"]], dtype=torch.long)
            if data.edge_index.numel() == 0:
                data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)

            split = meta["split"]
            if split == "train":
                train_graphs.append(data)
            elif split == "val":
                val_graphs.append(data)
            elif split == "test":
                test_graphs.append(data)
        except Exception as e:
            logger.warning("Failed loading graph %s: %s", pt_file.name, e)

    return train_graphs, val_graphs, test_graphs

