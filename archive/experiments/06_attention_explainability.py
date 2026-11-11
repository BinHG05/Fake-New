"""
Generate attention-based case studies from the current GAT checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.paper_experiments.common import DEFAULT_CHECKPOINT, DEFAULT_GRAPH_DIR, DEFAULT_METADATA_PATH, load_binary_metadata, load_test_graphs


REPORT_PATH = PROJECT_ROOT / "results" / "explainability_report.json"
FIGURE_PATH = PROJECT_ROOT / "results" / "paper_figures" / "attention_case_studies.png"


def lazy_import_torch():
    import torch

    return torch


def load_model_with_attention(checkpoint_path: Path, device: str):
    torch = lazy_import_torch()
    from src.training.train_multimodal_gnn import MultimodalCascadeGNN

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_dim = state_dict.get("fusion.image_proj.weight", torch.zeros(1, 1025)).size(1)
    model = MultimodalCascadeGNN(
        input_dim=768 + image_dim,
        text_dim=768,
        image_dim=image_dim,
        num_classes=state_dict.get("classifier.3.bias", torch.zeros(2)).size(0),
        gnn_type="gat",
        fusion_type="cross_attention",
        learn_missing_image_emb=("missing_image_emb" in state_dict),
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def top_attention_comments(model, graph, node_texts: list[str], device: str):
    torch = lazy_import_torch()
    graph = graph.to(device)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, attention_payload = model(graph.x, graph.edge_index, batch, return_attention=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    predicted = int(probs.argmax())
    confidence = float(probs.max())
    real_prob = float(probs[0])
    fake_prob = float(probs[1]) if probs.shape[0] > 1 else 1.0 - real_prob

    layers = attention_payload.get("layer_attentions") or []
    if not layers:
        return {
            "prediction": predicted,
            "confidence": confidence,
            "prob_real": real_prob,
            "prob_fake": fake_prob,
            "top_comments": [],
        }

    layer = layers[0]
    edge_index = layer["edge_index"].detach().cpu()
    alpha = layer["alpha"].detach().cpu()
    if alpha.dim() > 1:
        alpha = alpha.mean(dim=-1)

    scores: dict[int, list[float]] = {}
    for idx in range(edge_index.size(1)):
        dst = int(edge_index[1, idx].item())
        if dst == 0:
            continue
        scores.setdefault(dst, []).append(float(alpha[idx].item()))

    ranked = []
    for node_idx, values in scores.items():
        ranked.append(
            {
                "node_index": node_idx,
                "attention_score": float(sum(values) / len(values)),
                "text": (node_texts[node_idx] if node_idx < len(node_texts) else "")[:280],
            }
        )
    ranked.sort(key=lambda item: item["attention_score"], reverse=True)

    return {
        "prediction": predicted,
        "confidence": confidence,
        "prob_real": real_prob,
        "prob_fake": fake_prob,
        "top_comments": ranked[:5],
    }


def select_case_studies(graphs, metadata):
    buckets = {
        "correct_real": [],
        "correct_fake": [],
        "incorrect_high_conf": [],
    }
    for graph in graphs:
        record = metadata[str(graph.post_id)]["record"]
        yield graph, record


def plot_cases(cases: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, max(4, len(cases) * 3.2)))
    if len(cases) == 1:
        axes = [axes]

    for axis, case in zip(axes, cases):
        comments = case["top_comments"][:3]
        labels = [f"N{item['node_index']}" for item in comments] or ["No comments"]
        values = [item["attention_score"] * 100 for item in comments] or [0.0]
        axis.bar(labels, values, color="#8b5cf6")
        axis.set_ylim(0, max(100, max(values, default=0) * 1.2))
        axis.set_ylabel("Attention (%)")
        axis.set_title(case["title"])
        for idx, item in enumerate(comments):
            axis.text(idx, values[idx] + 1.5, item["text"][:90], ha="center", va="bottom", fontsize=8, rotation=0, wrap=True)

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    torch = lazy_import_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_with_attention(DEFAULT_CHECKPOINT, device)
    graphs = load_test_graphs(DEFAULT_GRAPH_DIR, DEFAULT_METADATA_PATH)
    metadata = load_binary_metadata(DEFAULT_METADATA_PATH)

    all_cases = []
    for graph in graphs:
        record = metadata[str(graph.post_id)]["record"]
        node_texts = [str(record.get("raw_text") or record.get("clean_text") or record.get("title") or "")]
        node_texts.extend(str(comment.get("text") or "") for comment in (record.get("cascade") or []))
        info = top_attention_comments(model, graph, node_texts, device)
        info.update(
            {
                "post_id": str(graph.post_id),
                "label": int(graph.y.item()),
                "correct": int(graph.y.item()) == info["prediction"],
                "title": node_texts[0][:140],
                "comment_count": max(int(graph.num_nodes) - 1, 0),
            }
        )
        all_cases.append(info)

    correct_real = [c for c in all_cases if c["correct"] and c["label"] == 0]
    correct_fake = [c for c in all_cases if c["correct"] and c["label"] == 1]
    incorrect_high_conf = [c for c in all_cases if not c["correct"]]
    correct_real.sort(key=lambda item: item["confidence"], reverse=True)
    correct_fake.sort(key=lambda item: item["confidence"], reverse=True)
    incorrect_high_conf.sort(key=lambda item: item["confidence"], reverse=True)

    selected = []
    if correct_real:
        selected.append({**correct_real[0], "case_type": "correct_real"})
    if correct_fake:
        selected.append({**correct_fake[0], "case_type": "correct_fake"})
    if incorrect_high_conf:
        selected.append({**incorrect_high_conf[0], "case_type": "incorrect_high_conf"})

    plot_cases(selected)

    report = {
        "checkpoint": DEFAULT_CHECKPOINT.name,
        "num_test_graphs": len(graphs),
        "selected_cases": selected,
        "figure": str(FIGURE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
