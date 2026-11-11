"""
TN4: Domain shift evaluation on live Reddit posts outside the Fakeddit training distribution.

This script fetches fresh posts from curated subreddits, labels them by source policy,
runs live inference with FakeNewsPredictor, and compares OOD performance against the
current in-distribution binary checkpoint.

Outputs:
- data/ood_test_set.jsonl
- results/domain_shift_report.json
- results/paper_figures/domain_shift_comparison.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

REAL_TARGETS = [("worldnews", 5), ("science", 5), ("neutralnews", 5)]
FAKE_TARGETS = [("TheOnion", 15)]
IN_DISTRIBUTION_ACCURACY = 0.875
IN_DISTRIBUTION_F1 = 0.847161572052402


def clear_proxy_env() -> None:
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_session():
    import requests

    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "python:multimodal_gnn:1.0 (by /u/test)"})
    return session


def fetch_posts(session, subreddit: str, limit: int, ground_truth: str) -> list[dict]:
    url = f"https://old.reddit.com/r/{subreddit}/new.json?limit={limit}&raw_json=1"
    response = session.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    posts = []
    for child in payload.get("data", {}).get("children", []):
        data = child.get("data", {})
        permalink = data.get("permalink")
        post_id = data.get("id")
        title = data.get("title", "").strip()
        if not permalink or not post_id or not title:
            continue
        posts.append(
            {
                "id": post_id,
                "subreddit": subreddit,
                "ground_truth": ground_truth,
                "title": title,
                "url": f"https://www.reddit.com{permalink}",
                "created_utc": data.get("created_utc"),
                "num_comments_api": data.get("num_comments", 0),
            }
        )
    return posts


def select_ood_set(session) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[str] = set()

    for subreddit, target_count in REAL_TARGETS:
        fetched = fetch_posts(session, subreddit, max(target_count * 3, 10), "REAL")
        count = 0
        for post in fetched:
            if post["id"] in seen_ids:
                continue
            selected.append(post)
            seen_ids.add(post["id"])
            count += 1
            if count >= target_count:
                break
        if count < target_count:
            raise RuntimeError(f"Not enough REAL posts fetched from r/{subreddit}.")

    for subreddit, target_count in FAKE_TARGETS:
        fetched = fetch_posts(session, subreddit, max(target_count * 3, 20), "FAKE")
        count = 0
        for post in fetched:
            if post["id"] in seen_ids:
                continue
            selected.append(post)
            seen_ids.add(post["id"])
            count += 1
            if count >= target_count:
                break
        if count < target_count:
            raise RuntimeError(f"Not enough FAKE posts fetched from r/{subreddit}.")

    return selected


def compute_metrics(records: list[dict]) -> dict:
    y_true = [0 if row["ground_truth"] == "REAL" else 1 for row in records]
    y_pred = [0 if row["prediction"] == "REAL" else 1 for row in records]

    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    accuracy = correct / len(records) if records else 0.0

    tp = sum(int(t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum(int(t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    avg_conf = sum(float(row["confidence"]) for row in records) / len(records) if records else 0.0
    correct_conf = [float(row["confidence"]) for row in records if row["correct"]]
    wrong_conf = [float(row["confidence"]) for row in records if not row["correct"]]

    by_label = {}
    for label in ["REAL", "FAKE"]:
        subset = [row for row in records if row["ground_truth"] == label]
        if not subset:
            continue
        sub_acc = sum(int(row["correct"]) for row in subset) / len(subset)
        by_label[label] = {
            "count": len(subset),
            "accuracy": sub_acc,
            "avg_confidence": sum(float(row["confidence"]) for row in subset) / len(subset),
        }

    return {
        "count": len(records),
        "accuracy": accuracy,
        "f1_binary": f1,
        "avg_confidence": avg_conf,
        "avg_confidence_correct": sum(correct_conf) / len(correct_conf) if correct_conf else None,
        "avg_confidence_incorrect": sum(wrong_conf) / len(wrong_conf) if wrong_conf else None,
        "by_ground_truth": by_label,
    }


def plot_comparison(report: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Accuracy", "Binary F1", "Avg confidence"]
    in_distribution = [
        report["in_distribution"]["accuracy"] * 100,
        report["in_distribution"]["f1_binary"] * 100,
        report["in_distribution"]["avg_confidence"] * 100,
    ]
    ood = [
        report["ood"]["accuracy"] * 100,
        report["ood"]["f1_binary"] * 100,
        report["ood"]["avg_confidence"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, in_distribution, width, label="In-distribution", color="#5470C6")
    bars2 = ax.bar(x + width / 2, ood, width, label="OOD Reddit", color="#EE6666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Domain Shift: In-distribution vs Live Reddit OOD")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Reddit domain-shift evaluation.")
    parser.add_argument("--output-jsonl", type=Path, default=PROJECT_ROOT / "data" / "ood_test_set.jsonl")
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "results" / "domain_shift_report.json")
    parser.add_argument("--figure-path", type=Path, default=PROJECT_ROOT / "results" / "paper_figures" / "domain_shift_comparison.png")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    clear_proxy_env()
    session = build_session()

    from src.deployment.predictor import FakeNewsPredictor

    posts = select_ood_set(session)
    predictor = FakeNewsPredictor(device=args.device)

    enriched = []
    for index, post in enumerate(posts, start=1):
        result = predictor.predict_url(post["url"])
        enriched_row = {
            **post,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "warnings": result.get("warnings", []),
            "graph_info": result.get("graph_info", {}),
            "model_info": result.get("model_info", {}),
            "post_info": result.get("post_info", {}),
        }
        enriched_row["correct"] = enriched_row["prediction"] == enriched_row["ground_truth"]
        enriched.append(enriched_row)
        time.sleep(args.sleep_seconds)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ood_metrics = compute_metrics(enriched)
    in_distribution = {
        "accuracy": IN_DISTRIBUTION_ACCURACY,
        "f1_binary": IN_DISTRIBUTION_F1,
        "avg_confidence": 0.9113966791379836,
    }

    report = {
        "dataset_policy": {
            "real_subreddits": REAL_TARGETS,
            "fake_subreddits": FAKE_TARGETS,
            "note": "REAL is sourced from news/science subreddits; FAKE is sourced from satire subreddit r/TheOnion.",
        },
        "in_distribution": in_distribution,
        "ood": ood_metrics,
        "delta_accuracy_pp": (ood_metrics["accuracy"] - in_distribution["accuracy"]) * 100,
        "delta_f1_pp": (ood_metrics["f1_binary"] - in_distribution["f1_binary"]) * 100,
        "samples": [
            {
                "id": row["id"],
                "subreddit": row["subreddit"],
                "ground_truth": row["ground_truth"],
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "correct": row["correct"],
                "title": row["title"],
                "num_nodes": row["graph_info"].get("num_nodes"),
                "has_image": row["post_info"].get("has_image"),
            }
            for row in enriched
        ],
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_comparison(report, args.figure_path)
    print(json.dumps({"ood": ood_metrics, "delta_accuracy_pp": report["delta_accuracy_pp"]}, indent=2))


if __name__ == "__main__":
    main()
