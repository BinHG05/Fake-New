"""
TN2: Error analysis for the binary multimodal cascade checkpoint.

Outputs:
- results/paper_figures/confusion_matrix.png
- results/paper_figures/accuracy_by_graph_size.png
- results/paper_figures/confidence_reliability.png
- results/error_analysis_report.md
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
    branching_factor,
    comment_count,
    graph_depth,
    load_binary_metadata,
    load_model,
    load_test_graphs,
    predict_graphs,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "error_analysis_report.md"
DEFAULT_JSON_PATH = PROJECT_ROOT / "results" / "error_analysis_report.json"


def graph_size_bucket(num_comments: int) -> str:
    if num_comments == 0:
        return "0"
    if num_comments <= 10:
        return "1-10"
    return ">10"


def confidence_band(confidence: float) -> str:
    if confidence > 0.7:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def plot_confusion(predictions, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true = [p.y_true for p in predictions]
    y_pred = [p.y_pred for p in predictions]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=axes[0])
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd", xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ax=axes[1])
    axes[1].set_title("Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.suptitle("Binary Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_graph_size(bucket_stats: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_reliability(band_stats: dict[str, dict[str, float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error analysis on the binary test split.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    graphs = load_test_graphs(args.graph_dir, args.metadata)
    metadata = load_binary_metadata(args.metadata)
    model = load_model(args.checkpoint, device=args.device)
    predictions = predict_graphs(model, graphs, device=args.device)

    pred_by_id = {p.post_id: p for p in predictions}
    graph_by_id = {str(g.post_id): g for g in graphs}

    y_true = np.array([p.y_true for p in predictions], dtype=int)
    y_pred = np.array([p.y_pred for p in predictions], dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision_real = tn / (tn + fn) if (tn + fn) else 0.0
    recall_real = tn / (tn + fp) if (tn + fp) else 0.0
    precision_fake = tp / (tp + fp) if (tp + fp) else 0.0
    recall_fake = tp / (tp + fn) if (tp + fn) else 0.0

    bucket_stats: dict[str, dict[str, float]] = {}
    for bucket in ["0", "1-10", ">10"]:
        bucket_preds = [p for p in predictions if graph_size_bucket(comment_count(graph_by_id[p.post_id])) == bucket]
        bucket_stats[bucket] = {
            "count": len(bucket_preds),
            "accuracy": accuracy(bucket_preds),
        }

    band_stats: dict[str, dict[str, float]] = {}
    for band in ["low", "medium", "high"]:
        band_preds = [p for p in predictions if confidence_band(p.confidence) == band]
        band_stats[band] = {
            "count": len(band_preds),
            "accuracy": accuracy(band_preds),
            "avg_confidence": float(np.mean([p.confidence for p in band_preds])) if band_preds else 0.0,
        }

    false_positives = []
    false_negatives = []
    for prediction in predictions:
        if prediction.y_true == 0 and prediction.y_pred == 1:
            false_positives.append(prediction)
        elif prediction.y_true == 1 and prediction.y_pred == 0:
            false_negatives.append(prediction)

    false_positives = sorted(false_positives, key=lambda item: item.prob_fake, reverse=True)[:5]
    false_negatives = sorted(false_negatives, key=lambda item: item.prob_real, reverse=True)[:5]

    def build_case_rows(items, expected_label: str, predicted_label: str) -> list[dict]:
        rows = []
        for item in items:
            record = metadata[item.post_id]["record"]
            graph = graph_by_id[item.post_id]
            raw_text = (
                record.get("raw_text")
                or record.get("clean_text")
                or record.get("title")
                or ""
            ).strip().replace("\n", " ")
            rows.append(
                {
                    "id": item.post_id,
                    "expected": expected_label,
                    "predicted": predicted_label,
                    "confidence": round(item.confidence, 4),
                    "prob_real": round(item.prob_real, 4),
                    "prob_fake": round(item.prob_fake, 4),
                    "comments": comment_count(graph),
                    "depth": graph_depth(graph),
                    "branching_factor": round(branching_factor(graph), 3),
                    "average_degree": round(average_degree(graph), 3),
                    "text_preview": raw_text[:180],
                }
            )
        return rows

    fp_rows = build_case_rows(false_positives, "REAL", "FAKE")
    fn_rows = build_case_rows(false_negatives, "FAKE", "REAL")

    report = {
        "overall_accuracy": accuracy(predictions),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "class_metrics": {
            "real": {"precision": precision_real, "recall": recall_real},
            "fake": {"precision": precision_fake, "recall": recall_fake},
        },
        "accuracy_by_graph_size": bucket_stats,
        "confidence_bands": band_stats,
        "false_positive_cases": fp_rows,
        "false_negative_cases": fn_rows,
    }

    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.skip_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion(predictions, args.output_dir / "confusion_matrix.png")
        plot_accuracy_by_graph_size(bucket_stats, args.output_dir / "accuracy_by_graph_size.png")
        plot_confidence_reliability(band_stats, args.output_dir / "confidence_reliability.png")

    lines = [
        "# Error Analysis Report",
        "",
        f"- Overall accuracy: **{report['overall_accuracy'] * 100:.2f}%**",
        f"- REAL precision/recall: **{precision_real * 100:.2f}% / {recall_real * 100:.2f}%**",
        f"- FAKE precision/recall: **{precision_fake * 100:.2f}% / {recall_fake * 100:.2f}%**",
        "",
        "## Accuracy by Graph Size",
        "",
    ]
    for bucket in ["0", "1-10", ">10"]:
        stat = bucket_stats[bucket]
        lines.append(f"- `{bucket}` comments: {stat['accuracy'] * 100:.2f}% accuracy on {stat['count']} samples")

    lines.extend(["", "## Confidence Bands", ""])
    for band in ["low", "medium", "high"]:
        stat = band_stats[band]
        lines.append(
            f"- `{band}`: {stat['accuracy'] * 100:.2f}% accuracy, avg confidence {stat['avg_confidence'] * 100:.2f}%, count {stat['count']}"
        )

    lines.extend(["", "## Top False Positives", ""])
    for row in fp_rows:
        lines.append(
            f"- `{row['id']}` | conf={row['confidence']:.4f} | comments={row['comments']} | depth={row['depth']} | text=`{row['text_preview']}`"
        )

    lines.extend(["", "## Top False Negatives", ""])
    for row in fn_rows:
        lines.append(
            f"- `{row['id']}` | conf={row['confidence']:.4f} | comments={row['comments']} | depth={row['depth']} | text=`{row['text_preview']}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The report is designed to answer whether the checkpoint is biased toward one class,",
            "whether larger cascades actually reduce error, and whether the confidence score is meaningful.",
            "",
            "```json",
            json.dumps(report, indent=2),
            "```",
            "",
        ]
    )
    args.report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
