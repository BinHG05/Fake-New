"""
Run refreshed multi-seed experiments for the 9,584-sample binary dataset.

This script locks the full-model training configuration to the rerun setup that
restores `Full > No-Graph`, then aggregates mean/std over multiple seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(r"D:\anacoda3\envs\multimodal_gnn\python.exe")
OUTPUT_DIR = PROJECT_ROOT / "reports" / "results" / "paper_experiments"


def parse_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    patterns = {
        "acc_bin": r"Binary Accuracy:\s+([0-9.]+)",
        "f1_bin": r"Binary F1:\s+([0-9.]+)",
        "threshold": r"Threshold:\s+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def build_command(python_exe: str, seed: int, save_dir: Path) -> list[str]:
    return [
        python_exe,
        "-u",
        "src/training/train_multimodal_gnn.py",
        "--graph_dir",
        "data/processed_graphs_v3",
        "--metadata",
        "data/all_merged_binary.jsonl",
        "--fusion_type",
        "cross_attention",
        "--gnn_type",
        "gat",
        "--epochs",
        "100",
        "--patience",
        "20",
        "--batch_size",
        "32",
        "--lr",
        "5e-5",
        "--weight_decay",
        "1e-3",
        "--seed",
        str(seed),
        "--augment_edge_drop",
        "0.1",
        "--augment_feature_mask",
        "0.1",
        "--lr_scheduler",
        "cosine",
        "--learn_missing_image_emb",
        "--save_dir",
        str(save_dir),
    ]


def run_seed(python_exe: str, seed: int) -> dict:
    save_dir = PROJECT_ROOT / "models" / "checkpoints" / f"rerun_full_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(python_exe, seed, save_dir)

    print(f"\n=== Seed {seed} ===")
    print(" ".join(cmd))
    process = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (process.stdout or "") + (process.stderr or "")
    metrics = parse_metrics(output)
    if process.returncode != 0 or "acc_bin" not in metrics:
        raise RuntimeError(f"Training failed for seed {seed}.\n{output}")
    return {
        "seed": seed,
        "acc_bin": metrics["acc_bin"],
        "f1_bin": metrics.get("f1_bin", 0.0),
        "threshold": metrics.get("threshold", 0.5),
        "save_dir": str(save_dir),
    }


def summarize(results: list[dict]) -> dict:
    accs = [row["acc_bin"] for row in results]
    f1s = [row["f1_bin"] for row in results]
    thresholds = [row["threshold"] for row in results]
    return {
        "acc_bin": {
            "mean": statistics.mean(accs),
            "std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "values": accs,
        },
        "f1_bin": {
            "mean": statistics.mean(f1s),
            "std": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "values": f1s,
        },
        "threshold": {
            "mean": statistics.mean(thresholds),
            "std": statistics.pstdev(thresholds) if len(thresholds) > 1 else 0.0,
            "values": thresholds,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run refreshed multi-seed full-model experiments.")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = [run_seed(args.python, seed) for seed in args.seeds]
    payload = {
        "config": {
            "graph_dir": "data/processed_graphs_v3",
            "metadata": "data/all_merged_binary.jsonl",
            "fusion_type": "cross_attention",
            "gnn_type": "gat",
            "epochs": 100,
            "patience": 20,
            "batch_size": 32,
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "augment_edge_drop": 0.1,
            "augment_feature_mask": 0.1,
            "lr_scheduler": "cosine",
            "learn_missing_image_emb": True,
        },
        "individual": results,
        "summary": summarize(results),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "multi_seed_refresh_20260505.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved multi-seed summary to: {output_path}")


if __name__ == "__main__":
    main()
