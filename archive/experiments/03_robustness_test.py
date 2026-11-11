"""
TN3: robustness analysis for the binary multimodal cascade checkpoint.

Outputs:
- results/paper_figures/robustness_missing_image.png
- results/paper_figures/robustness_sparse_cascade.png
- results/paper_figures/robustness_adversarial.png
- results/robustness_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.paper_experiments.common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GRAPH_DIR,
    DEFAULT_METADATA_PATH,
    accuracy,
    append_comment_nodes,
    binary_f1,
    comment_count,
    keep_root_plus_k_comments,
    load_model,
    load_test_graphs,
    predict_graphs,
    zero_image_features,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "robustness_report.json"


def plot_missing_image(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Full", "Missing Image"]
    acc = [metrics["full"]["accuracy"] * 100, metrics["missing_image"]["accuracy"] * 100]
    f1 = [metrics["full"]["f1"] * 100, metrics["missing_image"]["f1"] * 100]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#5470C6")
    bars2 = ax.bar(x + width / 2, f1, width, label="Binary F1", color="#91CC75")
    ax.set_title("Robustness to Missing Image Features")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sparse_cascade(metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Full", "Root+1", "Root+3"]
    scores = [metrics["full"]["accuracy"] * 100, metrics["root_plus_1"]["accuracy"] * 100, metrics["root_plus_3"]["accuracy"] * 100]
    counts = [metrics["full"]["count"], metrics["root_plus_1"]["count"], metrics["root_plus_3"]["count"]]

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    bars = ax1.bar(labels, scores, color=["#5470C6", "#FAC858", "#73C0DE"], edgecolor="white")
    ax1.set_title("Robustness to Sparse Cascade")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.2f}", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(labels, counts, color="#EE6666", marker="o", linewidth=2)
    ax2.set_ylabel("Sample count")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial(metrics: dict[str, float], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Original REAL kept", "Flipped to FAKE"]
    values = [metrics["kept_real_rate"] * 100, metrics["flip_rate"] * 100]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#91CC75", "#EE6666"], edgecolor="white")
    ax.set_title("Adversarial Comment Injection")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness tests on the binary checkpoint.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adversarial-sample-size", type=int, default=50)
    parser.add_argument("--adversarial-num-comments", type=int, default=5)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    graphs = load_test_graphs(args.graph_dir, args.metadata)
    model = load_model(args.checkpoint, device=args.device)

    full_predictions = predict_graphs(model, graphs, device=args.device)

    missing_image_graphs = [zero_image_features(graph) for graph in graphs]
    missing_image_predictions = predict_graphs(model, missing_image_graphs, device=args.device)

    sparse_source_graphs = [graph for graph in graphs if comment_count(graph) > 5]
    root_plus_1_graphs = [keep_root_plus_k_comments(graph, 1, seed=args.seed) for graph in sparse_source_graphs]
    root_plus_3_graphs = [keep_root_plus_k_comments(graph, 3, seed=args.seed) for graph in sparse_source_graphs]
    sparse_pred_1 = predict_graphs(model, root_plus_1_graphs, device=args.device)
    sparse_pred_3 = predict_graphs(model, root_plus_3_graphs, device=args.device)
    sparse_full_pred = predict_graphs(model, sparse_source_graphs, device=args.device)

    fake_comment_pool = []
    for graph in graphs:
        if int(graph.y.item()) == 1 and int(graph.num_nodes) > 1:
            fake_comment_pool.append(graph.x[1:, :].detach().cpu())
    if fake_comment_pool:
        fake_comment_prototype = np.concatenate([tensor.numpy() for tensor in fake_comment_pool], axis=0).mean(axis=0)
    else:
        fake_comment_prototype = graphs[0].x[0, :].detach().cpu().numpy()

    torch = __import__("torch")
    fake_comment_prototype = torch.tensor(fake_comment_prototype, dtype=graphs[0].x.dtype)

    real_graphs = [graph for graph in graphs if int(graph.y.item()) == 0]
    original_real_predictions = [p for p in full_predictions if p.y_true == 0]
    real_pred_by_id = {p.post_id: p for p in original_real_predictions}
    candidate_real_graphs = [graph for graph in real_graphs if real_pred_by_id[str(graph.post_id)].y_pred == 0]
    candidate_real_graphs = candidate_real_graphs[: args.adversarial_sample_size]

    adversarial_graphs = [
        append_comment_nodes(graph, fake_comment_prototype, num_new_nodes=args.adversarial_num_comments, attach_to_root=True)
        for graph in candidate_real_graphs
    ]
    adversarial_predictions = predict_graphs(model, adversarial_graphs, device=args.device)

    flip_count = sum(int(pred.y_pred == 1) for pred in adversarial_predictions)
    total_adv = len(adversarial_predictions)

    report = {
        "missing_image": {
            "full": {"accuracy": accuracy(full_predictions), "f1": binary_f1(full_predictions)},
            "missing_image": {"accuracy": accuracy(missing_image_predictions), "f1": binary_f1(missing_image_predictions)},
            "delta_accuracy_pp": (accuracy(missing_image_predictions) - accuracy(full_predictions)) * 100,
            "delta_f1_pp": (binary_f1(missing_image_predictions) - binary_f1(full_predictions)) * 100,
        },
        "sparse_cascade": {
            "full": {"accuracy": accuracy(sparse_full_pred), "f1": binary_f1(sparse_full_pred), "count": len(sparse_full_pred)},
            "root_plus_1": {"accuracy": accuracy(sparse_pred_1), "f1": binary_f1(sparse_pred_1), "count": len(sparse_pred_1)},
            "root_plus_3": {"accuracy": accuracy(sparse_pred_3), "f1": binary_f1(sparse_pred_3), "count": len(sparse_pred_3)},
            "delta_root_plus_1_pp": (accuracy(sparse_pred_1) - accuracy(sparse_full_pred)) * 100,
            "delta_root_plus_3_pp": (accuracy(sparse_pred_3) - accuracy(sparse_full_pred)) * 100,
        },
        "adversarial_injection": {
            "surrogate": "feature-space fake-comment prototype averaged from FAKE test comment nodes",
            "sample_size": total_adv,
            "num_injected_comments": args.adversarial_num_comments,
            "flip_count": flip_count,
            "flip_rate": (flip_count / total_adv) if total_adv else 0.0,
            "kept_real_rate": ((total_adv - flip_count) / total_adv) if total_adv else 0.0,
        },
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.skip_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_missing_image(report["missing_image"], args.output_dir / "robustness_missing_image.png")
        plot_sparse_cascade(report["sparse_cascade"], args.output_dir / "robustness_sparse_cascade.png")
        plot_adversarial(report["adversarial_injection"], args.output_dir / "robustness_adversarial.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
