"""
Shared helpers for final paper experiments.

The goal is to keep TN5 / TN2 / TN3 / TN4 aligned on one evaluation path:
- same metadata split logic
- same checkpoint loading logic
- same binary prediction format
- same graph mutation helpers
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "processed_graphs_v3"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "all_merged_binary.jsonl"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "checkpoints" / "best_multimodal_cascade_binary_cross_attention.pt"
)


@dataclass
class BinaryPrediction:
    post_id: str
    y_true: int
    y_pred: int
    prob_real: float
    prob_fake: float
    confidence: float
    num_nodes: int
    num_edges: int


def _lazy_import_torch():
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    return torch, Data, DataLoader


def load_binary_metadata(metadata_path: Path = DEFAULT_METADATA_PATH) -> dict[str, dict[str, Any]]:
    from src.data.label_utils import BINARY_LABEL_MAP, derive_binary_label

    id_to_meta: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                binary_label, _ = derive_binary_label(record)
            except ValueError:
                continue
            id_to_meta[str(record["id"])] = {
                "label": BINARY_LABEL_MAP[binary_label],
                "label_name": binary_label,
                "split": record.get("split"),
                "record": record,
            }
    return id_to_meta


def load_test_graphs(
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> list[Any]:
    torch, Data, _ = _lazy_import_torch()
    torch.serialization.add_safe_globals([Data])
    metadata = load_binary_metadata(metadata_path)

    graphs: list[Any] = []
    for pt_file in sorted(graph_dir.glob("*.pt")):
        post_id = pt_file.stem
        meta = metadata.get(post_id)
        if not meta or meta["split"] != "test":
            continue

        graph = torch.load(pt_file, weights_only=False)
        graph.post_id = post_id
        graph.y = torch.tensor([meta["label"]], dtype=torch.long)
        if graph.edge_index.numel() == 0:
            graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graphs.append(graph)
    return graphs


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
    torch, _, _ = _lazy_import_torch()
    from src.training.train_gnn import CascadeGNN
    from src.training.train_multimodal_gnn import MultimodalCascadeGNN

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_multimodal = any("fusion" in key for key in state_dict.keys())
    num_classes = state_dict.get("classifier.3.bias", torch.zeros(2)).size(0)

    gnn_type = "sage"
    if any("att_src" in key for key in state_dict.keys()):
        gnn_type = "gat"
    elif any("convs.0.weight" in key for key in state_dict.keys()) and not any(
        "lin_l" in key for key in state_dict.keys()
    ):
        gnn_type = "gcn"

    if is_multimodal:
        fusion_dim = state_dict.get("fusion.text_proj.weight", torch.zeros(256, 768)).size(0)
        image_dim = state_dict.get("fusion.image_proj.weight", torch.zeros(1, 1025)).size(1)
        learn_missing_image_emb = "missing_image_emb" in state_dict
        num_gnn_layers = len([key for key in state_dict.keys() if key.startswith("convs.") and key.endswith(".att")])
        if num_gnn_layers == 0:
            num_gnn_layers = 3
        model = MultimodalCascadeGNN(
            input_dim=768 + image_dim,
            text_dim=768,
            image_dim=image_dim,
            fusion_dim=fusion_dim,
            hidden_dim=fusion_dim,
            num_classes=num_classes,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
            fusion_type="cross_attention",
            learn_missing_image_emb=learn_missing_image_emb,
        )
    else:
        model = CascadeGNN(
            input_dim=768,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=0.3,
            gnn_type=gnn_type,
            num_layers=2,
        )

    model.load_state_dict(state_dict)
    return model.to(device).eval()


def predict_graphs(
    model,
    graphs: list[Any],
    device: str = "cpu",
    batch_size: int = 32,
    ablation: str | None = None,
) -> list[BinaryPrediction]:
    torch, _, DataLoader = _lazy_import_torch()
    import torch.nn.functional as F

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    predictions: list[BinaryPrediction] = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if ablation is None:
                logits = model(batch.x, batch.edge_index, batch.batch)
            else:
                logits = model(batch.x, batch.edge_index, batch.batch, ablation=ablation)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            y_true = batch.y.detach().cpu().numpy()
            y_pred = probs.argmax(axis=-1)

            batch_size_actual = len(y_pred)
            batch_graphs = graphs[offset : offset + batch_size_actual]
            offset += batch_size_actual

            for index, graph in enumerate(batch_graphs):
                prob_real = float(probs[index][0])
                prob_fake = float(probs[index][1]) if probs.shape[1] > 1 else 1.0 - prob_real
                predictions.append(
                    BinaryPrediction(
                        post_id=str(getattr(graph, "post_id", f"graph_{offset}_{index}")),
                        y_true=int(y_true[index]),
                        y_pred=int(y_pred[index]),
                        prob_real=prob_real,
                        prob_fake=prob_fake,
                        confidence=max(prob_real, prob_fake),
                        num_nodes=int(graph.num_nodes),
                        num_edges=int(graph.edge_index.size(1)),
                    )
                )

    return predictions


def clone_graph(graph):
    return copy.deepcopy(graph)


def keep_root_only(graph):
    torch, _, _ = _lazy_import_torch()
    clone = clone_graph(graph)
    clone.x = clone.x[:1].clone()
    clone.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    clone.num_nodes = int(clone.x.size(0))
    return clone


def keep_root_plus_k_comments(graph, k: int, seed: int = 42):
    torch, _, _ = _lazy_import_torch()
    rng = np.random.default_rng(seed)
    clone = clone_graph(graph)
    num_nodes = int(clone.num_nodes)
    if num_nodes <= 1:
        clone.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return clone

    keep_count = min(max(k, 0), num_nodes - 1)
    comment_indices = np.arange(1, num_nodes)
    if keep_count < len(comment_indices):
        sampled = np.sort(rng.choice(comment_indices, size=keep_count, replace=False))
    else:
        sampled = comment_indices

    keep_nodes = np.concatenate(([0], sampled))
    node_map = {int(old): new for new, old in enumerate(keep_nodes.tolist())}

    clone.x = clone.x[keep_nodes].clone()
    edges = clone.edge_index.detach().cpu().numpy().T
    new_edges: list[list[int]] = []
    for src, dst in edges:
        src_i = int(src)
        dst_i = int(dst)
        if src_i in node_map and dst_i in node_map:
            new_edges.append([node_map[src_i], node_map[dst_i]])

    if not new_edges:
        clone.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        clone.edge_index = torch.tensor(new_edges, dtype=torch.long).T.contiguous()
    clone.num_nodes = int(clone.x.size(0))
    return clone


def inject_text_signal_to_root(graph, scale: float = 0.15):
    clone = clone_graph(graph)
    if clone.x.size(0) == 0:
        return clone
    clone.x[0, :768] = clone.x[0, :768] + scale
    return clone


def zero_image_features(graph, text_dim: int = 768):
    clone = clone_graph(graph)
    if clone.x.size(1) > text_dim:
        clone.x[:, text_dim:] = 0
    clone.num_nodes = int(clone.x.size(0))
    return clone


def append_comment_nodes(graph, node_feature, num_new_nodes: int = 5, attach_to_root: bool = True):
    torch, _, _ = _lazy_import_torch()
    clone = clone_graph(graph)
    if num_new_nodes <= 0:
        return clone

    feature = node_feature.detach().clone().to(dtype=clone.x.dtype, device=clone.x.device)
    if feature.dim() == 1:
        feature = feature.unsqueeze(0)
    feature = feature.repeat(num_new_nodes, 1)

    old_num_nodes = int(clone.x.size(0))
    clone.x = torch.cat([clone.x, feature], dim=0)

    new_edges = []
    if attach_to_root:
        for idx in range(num_new_nodes):
            new_node_id = old_num_nodes + idx
            new_edges.append([0, new_node_id])

    if new_edges:
        extra = torch.tensor(new_edges, dtype=torch.long, device=clone.edge_index.device).T.contiguous()
        clone.edge_index = torch.cat([clone.edge_index, extra], dim=1)

    clone.num_nodes = int(clone.x.size(0))
    return clone


def accuracy(predictions: list[BinaryPrediction]) -> float:
    if not predictions:
        return 0.0
    return float(sum(int(p.y_true == p.y_pred) for p in predictions) / len(predictions))


def binary_f1(predictions: list[BinaryPrediction]) -> float:
    y_true = np.array([p.y_true for p in predictions], dtype=int)
    y_pred = np.array([p.y_pred for p in predictions], dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def comment_count(graph) -> int:
    return max(int(graph.num_nodes) - 1, 0)


def graph_depth(graph) -> int:
    num_nodes = int(graph.num_nodes)
    if num_nodes <= 1:
        return 0
    adjacency = {idx: [] for idx in range(num_nodes)}
    for src, dst in graph.edge_index.detach().cpu().numpy().T:
        adjacency[int(src)].append(int(dst))

    visited = {0}
    frontier = [(0, 0)]
    max_depth = 0
    while frontier:
        node, depth = frontier.pop(0)
        max_depth = max(max_depth, depth)
        for child in adjacency.get(node, []):
            if child not in visited:
                visited.add(child)
                frontier.append((child, depth + 1))
    return max_depth


def branching_factor(graph) -> float:
    num_nodes = int(graph.num_nodes)
    if num_nodes <= 1:
        return 0.0
    outgoing = np.zeros(num_nodes, dtype=float)
    for src, _ in graph.edge_index.detach().cpu().numpy().T:
        outgoing[int(src)] += 1
    internal = outgoing[outgoing > 0]
    return float(internal.mean()) if internal.size else 0.0


def average_degree(graph) -> float:
    num_nodes = int(graph.num_nodes)
    if num_nodes == 0:
        return 0.0
    degree = np.zeros(num_nodes, dtype=float)
    for src, dst in graph.edge_index.detach().cpu().numpy().T:
        degree[int(src)] += 1
        degree[int(dst)] += 1
    return float(degree.mean())


def try_mann_whitney(sample_a: list[float], sample_b: list[float]) -> dict[str, float | None]:
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        return {"statistic": None, "p_value": None}

    if not sample_a or not sample_b:
        return {"statistic": None, "p_value": None}

    result = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
