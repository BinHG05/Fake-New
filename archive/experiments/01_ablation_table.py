"""
Build the definitive ablation package for the final paper/defense.

This script prefers loading metrics from a JSON file when available, but it can
also fall back to the validated numbers already captured in the repo's paper
artifacts. That makes it usable even when the original `all_results_*.json`
file is no longer present locally.

Outputs:
- results/paper_figures/ablation_definitive.png
- results/paper_figures/ablation_definitive.tex
- results/paper_figures/ablation_definitive_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_figures"

FALLBACK_RESULTS = [
    {
        "key": "phase3_text_only",
        "label": "T",
        "display_name": "Text-only",
        "family": "Phase 3",
        "binary_acc": 52.50,
        "binary_f1": 56.11,
        "question": "Baseline thuan NLP",
    },
    {
        "key": "phase3_graph_gat",
        "label": "T+G",
        "display_name": "Text + Graph (GAT)",
        "family": "Phase 3",
        "binary_acc": 63.93,
        "binary_f1": 49.75,
        "question": "Graph co giup khong?",
    },
    {
        "key": "phase3_image_only",
        "label": "I",
        "display_name": "Image-only",
        "family": "Phase 3",
        "binary_acc": 60.71,
        "binary_f1": 0.00,
        "question": "Image mot minh ra sao?",
    },
    {
        "key": "phase4_text_only",
        "label": "T",
        "display_name": "Text-only ablation",
        "family": "Phase 4",
        "binary_acc": 43.93,
        "binary_f1": 49.84,
        "question": "Neu cat image va graph thi con gi?",
    },
    {
        "key": "phase4_image_only",
        "label": "I",
        "display_name": "Image-only ablation",
        "family": "Phase 4",
        "binary_acc": 42.14,
        "binary_f1": 57.14,
        "question": "Neu cat text va graph thi con gi?",
    },
    {
        "key": "phase4_no_graph",
        "label": "T+I",
        "display_name": "Text + Image (No Graph)",
        "family": "Phase 4",
        "binary_acc": 46.07,
        "binary_f1": 58.40,
        "question": "Image co giup khi khong co cascade khong?",
    },
    {
        "key": "phase4_gated_fusion",
        "label": "CrossAtt vs Gated",
        "display_name": "Gated Fusion",
        "family": "Phase 4",
        "binary_acc": 46.07,
        "binary_f1": 55.98,
        "question": "Fusion gated so voi attention ra sao?",
    },
    {
        "key": "phase4_full_crossatt",
        "label": "T+I+G",
        "display_name": "Full Model (Cross-Attention)",
        "family": "Phase 4",
        "binary_acc": 66.43,
        "binary_f1": 56.88,
        "question": "Ket hop ca 3 co tot nhat khong?",
    },
]


def load_results(results_path: Path | None) -> list[dict]:
    if results_path is None:
        return FALLBACK_RESULTS

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    records: list[dict] = []

    baselines = payload.get("baselines", {})
    ablation = payload.get("ablation", {})

    mapping = [
        ("phase3_text_only", "T", "Text-only", "Phase 3", baselines.get("text_only")),
        ("phase3_graph_gat", "T+G", "Text + Graph (GAT)", "Phase 3", baselines.get("graph_gat")),
        ("phase3_image_only", "I", "Image-only", "Phase 3", baselines.get("image_only")),
        ("phase4_text_only", "T", "Text-only ablation", "Phase 4", ablation.get("text_only")),
        ("phase4_image_only", "I", "Image-only ablation", "Phase 4", ablation.get("image_only")),
        ("phase4_no_graph", "T+I", "Text + Image (No Graph)", "Phase 4", ablation.get("no_graph")),
        ("phase4_gated_fusion", "CrossAtt vs Gated", "Gated Fusion", "Phase 4", ablation.get("gated_fusion")),
        ("phase4_full_crossatt", "T+I+G", "Full Model (Cross-Attention)", "Phase 4", ablation.get("full_model")),
    ]

    for key, label, display_name, family, metrics in mapping:
        if not metrics:
            continue
        records.append(
            {
                "key": key,
                "label": label,
                "display_name": display_name,
                "family": family,
                "binary_acc": round(float(metrics.get("acc_bin", 0)) * 100, 2),
                "binary_f1": round(float(metrics.get("f1_bin", 0)) * 100, 2),
                "question": "",
            }
        )

    return records or FALLBACK_RESULTS


def compute_summary(records: list[dict]) -> dict:
    by_key = {record["key"]: record for record in records}
    full = by_key["phase4_full_crossatt"]
    graph = by_key["phase3_graph_gat"]
    no_graph = by_key["phase4_no_graph"]
    text_only = by_key["phase4_text_only"]
    gated = by_key["phase4_gated_fusion"]

    return {
        "full_vs_graph": round(full["binary_acc"] - graph["binary_acc"], 2),
        "full_vs_no_graph": round(full["binary_acc"] - no_graph["binary_acc"], 2),
        "full_vs_text_only": round(full["binary_acc"] - text_only["binary_acc"], 2),
        "crossatt_vs_gated": round(full["binary_acc"] - gated["binary_acc"], 2),
    }


def save_plot(records: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(records))
    acc = [record["binary_acc"] for record in records]
    f1 = [record["binary_f1"] for record in records]
    labels = [record["display_name"] for record in records]
    families = [record["family"] for record in records]

    colors = ["#5470C6" if family == "Phase 3" else "#91CC75" for family in families]
    width = 0.36

    fig, ax = plt.subplots(figsize=(16, 7))
    bars_acc = ax.bar(x - width / 2, acc, width, label="Binary Accuracy", color=colors, edgecolor="white")
    bars_f1 = ax.bar(x + width / 2, f1, width, label="Binary F1", color="#FAC858", edgecolor="white")

    ax.set_title("Definitive Ablation for Final Defense", fontsize=16, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    for idx, family in enumerate(families):
        ax.text(idx, 77.0, family, ha="center", va="bottom", fontsize=9, color="#444444")

    for bars in (bars_acc, bars_f1):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_latex(records: list[dict], summary: dict, output_path: Path) -> None:
    no_graph_delta = summary["full_vs_no_graph"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Binary ablation package used for the final defense narrative.}",
        r"\label{tab:ablation-definitive}",
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"\textbf{Config} & \textbf{Model} & \textbf{Acc (\%)} & \textbf{F1 (\%)} \\",
        r"\hline",
    ]

    for record in records:
        model = record["display_name"].replace("&", r"\&")
        lines.append(
            f'{record["label"]} & {model} & {record["binary_acc"]:.2f} & {record["binary_f1"]:.2f} \\\\'
        )

    lines.extend(
        [
            r"\hline",
            rf"\multicolumn{{4}}{{l}}{{Full vs Graph-only: +{summary['full_vs_graph']:.2f} pp Acc}} \\",
            rf"\multicolumn{{4}}{{l}}{{Full vs No-Graph: {no_graph_delta:+.2f} pp Acc}} \\",
            rf"\multicolumn{{4}}{{l}}{{Cross-Attention vs Gated: +{summary['crossatt_vs_gated']:.2f} pp Acc}} \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_summary(records: list[dict], summary: dict, output_path: Path) -> None:
    full = next(record for record in records if record["key"] == "phase4_full_crossatt")
    no_graph_delta = summary["full_vs_no_graph"]
    if no_graph_delta >= 0:
        no_graph_line = f"- Full model vs No-Graph: **+{no_graph_delta:.2f} pp** accuracy."
        takeaway = (
            "Cascade graph enrichment still adds measurable value on the refreshed binary split, "
            "while cross-attention remains stronger than the gated fusion variant."
        )
    else:
        no_graph_line = f"- Full model vs No-Graph: **{no_graph_delta:.2f} pp** accuracy."
        takeaway = (
            "On the refreshed binary split, the no-graph content path slightly outperforms the full graph path, "
            "so the contribution of cascade enrichment must be argued carefully and supported by TN5 rather than assumed."
        )

    lines = [
        "# Definitive Ablation Summary",
        "",
        f"- Full model: **{full['binary_acc']:.2f}%** accuracy, **{full['binary_f1']:.2f}%** F1.",
        f"- Full model vs Graph-only: **+{summary['full_vs_graph']:.2f} pp** accuracy.",
        no_graph_line,
        f"- Full model vs Text-only: **+{summary['full_vs_text_only']:.2f} pp** accuracy.",
        f"- Cross-Attention vs Gated fusion: **+{summary['crossatt_vs_gated']:.2f} pp** accuracy.",
        "",
        "## Scientific takeaway",
        "",
        takeaway,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final ablation package.")
    parser.add_argument("--results-json", type=Path, default=None, help="Optional all_results JSON file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = load_results(args.results_json)
    summary = compute_summary(records)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_plot(records, output_dir / "ablation_definitive.png")
    save_latex(records, summary, output_dir / "ablation_definitive.tex")
    save_summary(records, summary, output_dir / "ablation_definitive_summary.md")

    print("Saved definitive ablation package:")
    print(f"- {output_dir / 'ablation_definitive.png'}")
    print(f"- {output_dir / 'ablation_definitive.tex'}")
    print(f"- {output_dir / 'ablation_definitive_summary.md'}")


if __name__ == "__main__":
    main()
