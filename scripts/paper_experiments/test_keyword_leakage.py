"""
Mask keyword-bearing node embeddings and compare test accuracy/F1 to baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.paper_experiments.common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA_PATH,
    accuracy,
    binary_f1,
    clone_graph,
    load_binary_metadata,
    load_model,
    load_test_graphs,
    predict_graphs,
)


KEYWORDS = ("fake", "hoax", "debunk", "debunked", "misleading", "false", "source")
REPORT_PATH = PROJECT_ROOT / "results" / "keyword_leakage_report.json"


def node_texts_for_record(record: dict) -> list[str]:
    texts = [str(record.get("raw_text") or record.get("clean_text") or record.get("title") or "")]
    for comment in record.get("cascade", []) or []:
        texts.append(str(comment.get("text") or ""))
    return texts


def contains_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in KEYWORDS)


def mask_keyword_nodes(graph, texts: list[str]):
    clone = clone_graph(graph)
    mask_indices = [
        idx
        for idx, text in enumerate(texts[: int(clone.num_nodes)])
        if contains_keyword(text)
    ]
    if not mask_indices:
        return clone, 0

    mean_embedding = clone.x.mean(dim=0, keepdim=True)
    for idx in mask_indices:
        clone.x[idx] = mean_embedding[0]
    return clone, len(mask_indices)


def main():
    metadata = load_binary_metadata(Path(DEFAULT_METADATA_PATH))
    graphs = load_test_graphs()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = load_model(checkpoint_path=DEFAULT_CHECKPOINT, device=device)

    baseline = predict_graphs(model, graphs, device=device)
    masked_graphs = []
    affected_graphs = 0
    total_masked_nodes = 0

    for graph in graphs:
        record = metadata[str(graph.post_id)]["record"]
        masked_graph, count = mask_keyword_nodes(graph, node_texts_for_record(record))
        if count > 0:
            affected_graphs += 1
            total_masked_nodes += count
        masked_graphs.append(masked_graph)

    masked = predict_graphs(model, masked_graphs, device=device)

    report = {
        "keywords": list(KEYWORDS),
        "test_samples": len(graphs),
        "affected_graphs": affected_graphs,
        "total_masked_nodes": total_masked_nodes,
        "baseline": {
            "accuracy": accuracy(baseline),
            "f1_binary": binary_f1(baseline),
        },
        "masked": {
            "accuracy": accuracy(masked),
            "f1_binary": binary_f1(masked),
        },
    }
    report["delta"] = {
        "accuracy_pp": (report["masked"]["accuracy"] - report["baseline"]["accuracy"]) * 100.0,
        "f1_pp": (report["masked"]["f1_binary"] - report["baseline"]["f1_binary"]) * 100.0,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
