"""
Render refreshed paper figures from JSON/Markdown artifacts.

This script exists because the `multimodal_gnn` conda environment currently has
an incompatible matplotlib build. We therefore:
1. compute metrics with the ML environment
2. render figures with a separate Python that has a working matplotlib stack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_error_analysis(report_path: Path, output_dir: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    confusion = report["confusion"]
    cm = np.array(
        [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ],
        dtype=float,
    )
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=axes[1])
    axes[1].set_title("Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    fig.suptitle("Binary Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    bucket_stats = report["accuracy_by_graph_size"]
    order = ["0", "1-10", ">10"]
    scores = [bucket_stats.get(name, {}).get("accuracy", 0.0) * 100 for name in order]
    counts = [bucket_stats.get(name, {}).get("count", 0) for name in order]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    bars = ax1.bar(order, scores, color=["#73C0DE", "#91CC75", "#FAC858"], edgecolor="white")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Graph size bucket (comments)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Accuracy by Graph Size")
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, score + 1, f"{score:.2f}", ha="center", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(order, counts, color="#EE6666", marker="o", linewidth=2)
    ax2.set_ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_graph_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    band_stats = report["confidence_bands"]
    order = ["low", "medium", "high"]
    acc = [band_stats.get(name, {}).get("accuracy", 0.0) * 100 for name in order]
    avg_conf = [band_stats.get(name, {}).get("avg_confidence", 0.0) * 100 for name in order]
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#5470C6")
    bars2 = ax.bar(x + width / 2, avg_conf, width, label="Avg confidence", color="#91CC75")
    ax.set_xticks(x)
    ax.set_xticklabels([name.title() for name in order])
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Confidence Reliability by Band")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_robustness(report_path: Path, output_dir: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    missing = report["missing_image"]
    labels = ["Full", "Missing Image"]
    acc = [missing["full"]["accuracy"] * 100, missing["missing_image"]["accuracy"] * 100]
    f1 = [missing["full"]["f1"] * 100, missing["missing_image"]["f1"] * 100]
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
    plt.savefig(output_dir / "robustness_missing_image.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sparse = report["sparse_cascade"]
    labels = ["Full", "Root+1", "Root+3"]
    scores = [sparse["full"]["accuracy"] * 100, sparse["root_plus_1"]["accuracy"] * 100, sparse["root_plus_3"]["accuracy"] * 100]
    counts = [sparse["full"]["count"], sparse["root_plus_1"]["count"], sparse["root_plus_3"]["count"]]
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
    plt.savefig(output_dir / "robustness_sparse_cascade.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    adv = report["adversarial_injection"]
    labels = ["Original REAL kept", "Flipped to FAKE"]
    values = [adv["kept_real_rate"] * 100, adv["flip_rate"] * 100]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#91CC75", "#EE6666"], edgecolor="white")
    ax.set_title("Adversarial Comment Injection")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "robustness_adversarial.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_cascade(report_path: Path, output_dir: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    overall = report["overall"]
    labels = ["Accuracy", "Binary F1"]
    full_scores = [overall["with_cascade_accuracy"] * 100, overall["with_cascade_f1"] * 100]
    root_scores = [overall["root_only_accuracy"] * 100, overall["root_only_f1"] * 100]
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
    plt.savefig(output_dir / "cascade_vs_no_cascade.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    bucket_metrics = report["by_comment_bucket"]
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
    plt.savefig(output_dir / "accuracy_vs_cascade_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    real_stats = report["topology"]["real_means"]
    fake_stats = report["topology"]["fake_means"]
    keys = ["depth", "branching_factor", "average_degree"]
    titles = ["Graph Depth", "Branching Factor", "Average Degree"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, key, title in zip(axes, keys, titles):
        axis.bar(["REAL", "FAKE"], [real_stats[key], fake_stats[key]], color=["#73C0DE", "#EE6666"], edgecolor="white")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "topology_real_vs_fake.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render refreshed binary paper figures from JSON artifacts.")
    parser.add_argument("--error-json", type=Path, required=True)
    parser.add_argument("--robustness-json", type=Path, required=True)
    parser.add_argument("--cascade-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_error_analysis(args.error_json, args.output_dir)
    render_robustness(args.robustness_json, args.output_dir)
    render_cascade(args.cascade_json, args.output_dir)


if __name__ == "__main__":
    main()
