"""
TN5: quantify the contribution of comment cascade enrichment.

Outputs:
- results/paper_figures/cascade_vs_no_cascade.png
- results/paper_figures/accuracy_vs_cascade_size.png
- results/paper_figures/topology_real_vs_fake.png
- results/cascade_contribution_report.json
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
    average_degree,
    binary_f1,
    branching_factor,
    comment_count,
    graph_depth,
    keep_root_only,
    load_model,
    load_test_graphs,
    predict_graphs,
    try_mann_whitney,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "cascade_contribution_report.json"


def bucket_name(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    return ">20"


def plot_before_after(full_predictions, root_predictions, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Accuracy", "Binary F1"]
    full_scores = [accuracy(full_predictions) * 100, binary_f1(full_predictions) * 100]
    root_scores = [accuracy(root_predictions) * 100, binary_f1(root_predictions) * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_full = ax.bar(x - width / 2, full_scores, width, label="With Cascade", color="#5470C6")
    bars_root = ax.bar(x + width / 2, root_scores, width, label="Root Only", color="#EE6666")

    ax.set_title("Cascade Contribution: Before vs After")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (bars_full, bars_root):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_bucket(bucket_metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["0", "1-5", "6-20", ">20"]
    full_scores = [bucket_metrics.get(name, {}).get("full_accuracy", 0.0) * 100 for name in order]
    root_scores = [bucket_metrics.get(name, {}).get("root_accuracy", 0.0) * 100 for name in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(order, full_scores, marker="o", linewidth=2.2, color="#91CC75", label="With Cascade")
    ax.plot(order, root_scores, marker="o", linewidth=2.2, color="#EE6666", label="Root Only")

    ax.set_title("Accuracy vs Comment Cascade Size")
    ax.set_xlabel("Number of comments")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()

    for idx, label in enumerate(order):
        ax.text(idx, full_scores[idx] + 1.2, f"{full_scores[idx]:.1f}", ha="center", fontsize=8)
        ax.text(idx, root_scores[idx] - 4.0, f"{root_scores[idx]:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_topology(real_stats: dict[str, list[float]], fake_stats: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["depth", "branching_factor", "average_degree"]
    titles = ["Graph Depth", "Branching Factor", "Average Degree"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric, title in zip(axes, metrics, titles):
        axis.boxplot(
            [real_stats[metric], fake_stats[metric]],
            tick_labels=["REAL", "FAKE"],
            patch_artist=True,
            boxprops={"facecolor": "#73C0DE"},
            medianprops={"color": "#333333"},
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cascade contribution on the binary test set.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    graphs = load_test_graphs(args.graph_dir, args.metadata)
    model = load_model(args.checkpoint, device=args.device)

    root_only_graphs = [keep_root_only(graph) for graph in graphs]
    full_predictions = predict_graphs(model, graphs, device=args.device)
    root_predictions = predict_graphs(model, root_only_graphs, device=args.device)

    post_id_to_graph = {str(graph.post_id): graph for graph in graphs}
    full_by_id = {prediction.post_id: prediction for prediction in full_predictions}
    root_by_id = {prediction.post_id: prediction for prediction in root_predictions}

    bucket_metrics: dict[str, dict[str, float]] = {}
    for bucket in ["0", "1-5", "6-20", ">20"]:
        bucket_predictions_full = []
        bucket_predictions_root = []
        for post_id, graph in post_id_to_graph.items():
            if bucket_name(comment_count(graph)) != bucket:
                continue
            bucket_predictions_full.append(full_by_id[post_id])
            bucket_predictions_root.append(root_by_id[post_id])

        bucket_metrics[bucket] = {
            "count": len(bucket_predictions_full),
            "full_accuracy": accuracy(bucket_predictions_full),
            "root_accuracy": accuracy(bucket_predictions_root),
            "delta_accuracy_pp": (accuracy(bucket_predictions_full) - accuracy(bucket_predictions_root)) * 100,
        }

    real_topology = {"depth": [], "branching_factor": [], "average_degree": []}
    fake_topology = {"depth": [], "branching_factor": [], "average_degree": []}

    for graph in graphs:
        target = real_topology if int(graph.y.item()) == 0 else fake_topology
        target["depth"].append(graph_depth(graph))
        target["branching_factor"].append(branching_factor(graph))
        target["average_degree"].append(average_degree(graph))

    topology_tests = {
        "depth": try_mann_whitney(real_topology["depth"], fake_topology["depth"]),
        "branching_factor": try_mann_whitney(real_topology["branching_factor"], fake_topology["branching_factor"]),
        "average_degree": try_mann_whitney(real_topology["average_degree"], fake_topology["average_degree"]),
    }

    report = {
        "overall": {
            "with_cascade_accuracy": accuracy(full_predictions),
            "with_cascade_f1": binary_f1(full_predictions),
            "root_only_accuracy": accuracy(root_predictions),
            "root_only_f1": binary_f1(root_predictions),
            "delta_accuracy_pp": (accuracy(full_predictions) - accuracy(root_predictions)) * 100,
            "delta_f1_pp": (binary_f1(full_predictions) - binary_f1(root_predictions)) * 100,
        },
        "by_comment_bucket": bucket_metrics,
        "topology": {
            "real_means": {key: float(np.mean(values)) if values else 0.0 for key, values in real_topology.items()},
            "fake_means": {key: float(np.mean(values)) if values else 0.0 for key, values in fake_topology.items()},
            "mann_whitney": topology_tests,
        },
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.skip_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_before_after(full_predictions, root_predictions, args.output_dir / "cascade_vs_no_cascade.png")
        plot_accuracy_by_bucket(bucket_metrics, args.output_dir / "accuracy_vs_cascade_size.png")
        plot_topology(real_topology, fake_topology, args.output_dir / "topology_real_vs_fake.png")

    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
