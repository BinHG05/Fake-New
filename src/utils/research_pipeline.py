"""
Unified team-facing pipeline wrapper for the research workflow.

Stages:
1. Crawl Reddit data
2. Prepare unlabeled Reddit batch for Label Studio
3. Merge Label Studio export into labeled_master and refresh binary master
4. Enrich labeled data with comment trees and refresh binary enriched data
5. Build graphs (and optional multimodal graphs)
6. Train baseline / GNN / multimodal GNN
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def run_command(command: list[str], description: str) -> None:
    print(f"\n=== {description} ===")
    print("Command:", " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def cmd_crawl(args: argparse.Namespace) -> None:
    command = [PYTHON, "src/data/reddit_crawler.py", "--limit", str(args.limit)]
    if args.images_only:
        command.append("--images-only")
    if args.skip_comments:
        command.append("--skip-comments")
    if args.subreddits:
        command.extend(["--subreddits", *args.subreddits])
    run_command(command, "Step 1: Crawl Reddit")


def cmd_prepare_label(args: argparse.Namespace) -> None:
    command = [PYTHON, "src/utils/reddit_pipeline.py", "--input", args.input]
    if args.start is not None and args.count is not None:
        command.extend(["--start", str(args.start), "--count", str(args.count)])
    run_command(command, "Step 2: Prepare Reddit batch for Label Studio")


def cmd_merge_labels(args: argparse.Namespace) -> None:
    run_command(
        [
            PYTHON,
            "src/utils/merge_ls_export_by_id.py",
            "--input",
            args.input,
            "--output",
            args.master_output,
        ],
        "Step 3A: Merge Label Studio export into labeled_master",
    )
    run_command(
        [
            PYTHON,
            "src/utils/build_binary_labeled_master.py",
            "--input",
            args.master_output,
            "--output",
            args.binary_output,
            "--replace-label",
        ],
        "Step 3B: Refresh labeled_master_binary",
    )


def cmd_auto_label(args: argparse.Namespace) -> None:
    """Step 2.5: Auto-label unlabeled data using LLM (Groq by default)."""
    auto_labeler_script = PROJECT_ROOT / "src" / "utils" / "auto_labeler.py"
    command = [
        PYTHON, str(auto_labeler_script),
        "--input", args.input,
        "--output", args.output,
        "--method", args.method,
        "--mode", "binary",
        "--threshold", str(args.threshold),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.process_all:
        command.append("--all")
    run_command(command, f"Step 2.5: Auto-Label data using {args.method.upper()}")

    # After auto-labeling, merge into labeled_master and refresh binary
    run_command(
        [
            PYTHON,
            "src/utils/build_binary_labeled_master.py",
            "--input", args.output,
            "--output", args.binary_output,
            "--replace-label",
        ],
        "Step 2.5b: Refresh labeled_master_binary from auto-labeled data",
    )


def cmd_full_pipeline(args: argparse.Namespace) -> None:
    """Run the full end-to-end pipeline: Crawl → Auto-Label → Enrich → Build Graphs → Train."""
    print("\n" + "=" * 60)
    print("  🤖 FULL AUTO PIPELINE — End-to-End")
    print("=" * 60)

    # Step 1: Crawl
    crawl_cmd = [PYTHON, "src/data/reddit_crawler.py", "--limit", str(args.crawl_limit)]
    if args.images_only:
        crawl_cmd.append("--images-only")
    if args.subreddits:
        crawl_cmd.extend(["--subreddits", *args.subreddits])
    run_command(crawl_cmd, "Step 1: Crawl Reddit")

    # Step 2: Prepare batch (image + text preprocessing)
    run_command(
        [PYTHON, "src/utils/reddit_pipeline.py",
         "--input", "data/01_raw/reddit/reddit_realtime_data.jsonl"],
        "Step 2: Prepare batch for processing",
    )

    # Read tracker to get the latest clean batch
    import json
    tracker_path = os.path.join(PROJECT_ROOT, "data", "01_raw", "reddit", "processed_tracker.json")
    try:
        with open(tracker_path, "r", encoding="utf-8") as f:
            last_batch = json.load(f).get("last_batch")
        clean_batch_file = f"data/03_clean/Reddit/{last_batch}/Reddit/train.jsonl"
    except Exception as e:
        print(f"❌ Error reading tracker: {e}")
        return

    # Step 2.5: Auto-Label with LLM
    auto_label_output = f"data/03_clean/Reddit/{last_batch}/reddit_auto_labeled.jsonl"
    auto_labeler_script = PROJECT_ROOT / "src" / "utils" / "auto_labeler.py"
    
    auto_label_cmd = [
        PYTHON, str(auto_labeler_script),
        "--input", clean_batch_file,
        "--output", auto_label_output,
        "--method", args.llm,
        "--mode", "binary",
        "--threshold", str(args.confidence_threshold),
    ]
    if args.require_human_review:
        auto_label_cmd.append("--ls-predictions")
    
    run_command(auto_label_cmd, f"Step 2.5: Auto-Label with {args.llm.upper()} (threshold={args.confidence_threshold})")

    # Step 2.5b: Merge the auto-labeled data into labeled_master.jsonl safely
    master_output = "data/03_clean/Fakeddit/labeled_master.jsonl"
    run_command(
        [
            PYTHON, "src/utils/merge_ls_export_by_id.py",
            "--input", auto_label_output,
            "--output", master_output,
        ],
        "Step 2.5b: Merge auto-labeled data into Master JSONL",
    )

    # Step 2.5c: Build binary master from updated labeled_master.jsonl
    run_command(
        [
            PYTHON, "src/utils/build_binary_labeled_master.py",
            "--input", master_output,
            "--output", "data/03_clean/Fakeddit/labeled_master_binary.jsonl",
            "--replace-label",
        ],
        "Step 2.5c: Refresh labeled_master_binary",
    )

    # Step 4: Enrich with comment tree
    run_command(
        [
            PYTHON, "src/utils/enrich_master_with_comments.py",
            "--input", master_output,
            "--output", "data/reddit_enriched_data.jsonl",
            "--delay", str(args.enrich_delay),
        ],
        "Step 4: Enrich with comment trees",
    )
    run_command(
        [
            PYTHON, "src/utils/build_binary_labeled_master.py",
            "--input", "data/reddit_enriched_data.jsonl",
            "--output", "data/reddit_enriched_binary.jsonl",
            "--replace-label",
        ],
        "Step 4b: Refresh enriched binary",
    )

    # Step 5: Build Graphs (Multimodal)
    run_command(
        [
            PYTHON, "src/utils/build_cascade_graphs.py",
            "--input", "data/reddit_enriched_data.jsonl",
            "--output", "data/processed_graphs",
        ],
        "Step 5A: Build text cascade graphs",
    )
    run_command(
        [
            PYTHON, "src/utils/rebuild_graphs_with_images.py",
            "--metadata", "data/reddit_enriched_data.jsonl",
            "--graph_dir", "data/processed_graphs",
            "--output_dir", "data/processed_graphs_multimodal",
        ],
        "Step 5B: Build multimodal graphs",
    )

    print("\n" + "=" * 60)
    print("  ✅ DATA PREPARATION PIPELINE COMPLETED!")
    print("  Dữ liệu đã sẵn sàng! Bạn có thể chuyển sang Tab Training để tự train thủ công.")
    print("=" * 60)


def cmd_train_all(args: argparse.Namespace) -> None:
    """Train all models one by one."""
    print("\n" + "=" * 60)
    print("  🏆 TRAIN ALL MODELS SEQUENTIALLY")
    print("=" * 60)

    epochs = str(args.epochs)
    batch_size = str(args.batch_size)
    
    models = [
        ("Text Baseline", "src/training/train_baseline.py", ["--model", "text", "--epochs", epochs, "--batch_size", batch_size]),
        ("Image Baseline", "src/training/train_baseline.py", ["--model", "image", "--epochs", epochs, "--batch_size", batch_size]),
        ("Fusion Baseline", "src/training/train_baseline.py", ["--model", "fusion", "--epochs", epochs, "--batch_size", batch_size]),
        ("Graph GNN", "src/training/train_gnn.py", ["--epochs", epochs, "--batch_size", batch_size]),
        ("Multimodal GNN", "src/training/train_multimodal_gnn.py", ["--epochs", epochs, "--batch_size", batch_size]),
    ]

    for name, script, script_args in models:
        cmd = [PYTHON, script] + script_args
        run_command(cmd, f"Training {name}")

    print("\n" + "=" * 60)
    print("  ✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)


def cmd_enrich(args: argparse.Namespace) -> None:
    run_command(
        [
            PYTHON,
            "src/utils/enrich_master_with_comments.py",
            "--input",
            args.input,
            "--output",
            args.output,
            "--delay",
            str(args.delay),
        ]
        + (["--limit", str(args.limit)] if args.limit is not None else []),
        "Step 4A: Enrich labeled data with comment trees",
    )
    run_command(
        [
            PYTHON,
            "src/utils/build_binary_labeled_master.py",
            "--input",
            args.output,
            "--output",
            args.binary_output,
            "--replace-label",
        ],
        "Step 4B: Refresh reddit_enriched_binary",
    )


def cmd_build_graphs(args: argparse.Namespace) -> None:
    run_command(
        [
            PYTHON,
            "src/utils/build_cascade_graphs.py",
            "--input",
            args.input,
            "--output",
            args.output,
        ]
        + (["--force"] if args.force else [])
        + (["--limit", str(args.limit)] if args.limit is not None else []),
        "Step 5A: Build text cascade graphs",
    )

    if args.multimodal:
        command = [
            PYTHON,
            "src/utils/rebuild_graphs_with_images.py",
            "--metadata",
            args.input,
            "--graph_dir",
            args.output,
            "--output_dir",
            args.multimodal_output,
            "--batch_size",
            str(args.image_batch_size),
        ]
        run_command(command, "Step 5B: Build multimodal graphs")


def cmd_train(args: argparse.Namespace) -> None:
    graph_dir = args.graph_dir
    if args.model == "gnn" and graph_dir == "data/processed_graphs_multimodal":
        graph_dir = "data/processed_graphs"

    if args.model == "baseline_text":
        command = [
            PYTHON,
            "src/training/train_baseline.py",
            "--model_type",
            "text",
            "--label_mode",
            "binary",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--save_dir",
            args.save_dir,
        ]
    elif args.model == "baseline_image":
        command = [
            PYTHON,
            "src/training/train_baseline.py",
            "--model_type",
            "image",
            "--label_mode",
            "binary",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--save_dir",
            args.save_dir,
        ]
    elif args.model == "baseline_fusion":
        command = [
            PYTHON,
            "src/training/train_baseline.py",
            "--model_type",
            "fusion",
            "--label_mode",
            "binary",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--save_dir",
            args.save_dir,
        ]
    elif args.model == "gnn":
        command = [
            PYTHON,
            "src/training/train_gnn.py",
            "--graph_dir",
            graph_dir,
            "--metadata",
            args.metadata,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--save_dir",
            args.save_dir,
        ]
    else:
        command = [
            PYTHON,
            "src/training/train_multimodal_gnn.py",
            "--graph_dir",
            graph_dir,
            "--metadata",
            args.metadata,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--save_dir",
            args.save_dir,
        ]

    run_command(command, "Step 6: Train model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified team pipeline for the fake-news research workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl = subparsers.add_parser("crawl", help="Step 1: Crawl Reddit data")
    crawl.add_argument("--limit", type=int, default=25)
    crawl.add_argument("--images-only", action="store_true")
    crawl.add_argument("--skip-comments", action="store_true")
    crawl.add_argument("--subreddits", nargs="+", default=None)
    crawl.set_defaults(func=cmd_crawl)

    prepare = subparsers.add_parser("prepare-label", help="Step 2: Prepare Reddit batch for Label Studio")
    prepare.add_argument("--input", default="data/01_raw/reddit/reddit_realtime_data.jsonl")
    prepare.add_argument("--start", type=int, default=None)
    prepare.add_argument("--count", type=int, default=None)
    prepare.set_defaults(func=cmd_prepare_label)

    merge = subparsers.add_parser("merge-labels", help="Step 3: Merge Label Studio export and refresh binary master")
    merge.add_argument("--input", required=True, help="Path to Label Studio JSON export")
    merge.add_argument("--master-output", default="data/03_clean/Fakeddit/labeled_master.jsonl")
    merge.add_argument("--binary-output", default="data/03_clean/Fakeddit/labeled_master_binary.jsonl")
    merge.set_defaults(func=cmd_merge_labels)

    enrich = subparsers.add_parser("enrich", help="Step 4: Enrich master data with comment trees")
    enrich.add_argument("--input", default="data/03_clean/Fakeddit/labeled_master.jsonl")
    enrich.add_argument("--output", default="data/reddit_enriched_data.jsonl")
    enrich.add_argument("--binary-output", default="data/reddit_enriched_binary.jsonl")
    enrich.add_argument("--delay", type=float, default=2.0)
    enrich.add_argument("--limit", type=int, default=None)
    enrich.set_defaults(func=cmd_enrich)

    build = subparsers.add_parser("build-graphs", help="Step 5: Build graph artifacts")
    build.add_argument("--input", default="data/reddit_enriched_data.jsonl")
    build.add_argument("--output", default="data/processed_graphs")
    build.add_argument("--force", action="store_true")
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--multimodal", action="store_true")
    build.add_argument("--multimodal-output", default="data/processed_graphs_multimodal")
    build.add_argument("--image-batch-size", type=int, default=32)
    build.set_defaults(func=cmd_build_graphs)

    train = subparsers.add_parser("train", help="Step 6: Train a model")
    train.add_argument(
        "--model",
        required=True,
        choices=["baseline_text", "baseline_image", "baseline_fusion", "gnn", "multimodal_gnn"],
    )
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--save_dir", default="models/checkpoints/team_runs")
    train.add_argument("--metadata", default="data/reddit_enriched_binary.jsonl")
    train.add_argument("--graph_dir", default="data/processed_graphs_multimodal")
    train.set_defaults(func=cmd_train)

    # ── Auto-label (Step 2.5) ──────────────────────────────────
    auto = subparsers.add_parser("auto-label", help="Step 2.5: Auto-label data using LLM (Groq by default)")
    auto.add_argument("--input", default="data/01_raw/reddit/reddit_realtime_data.jsonl")
    auto.add_argument("--output", default="data/03_clean/Fakeddit/labeled_master.jsonl")
    auto.add_argument("--binary-output", default="data/03_clean/Fakeddit/labeled_master_binary.jsonl")
    auto.add_argument("--method", choices=["groq", "clip", "text"], default="groq",
                      help="LLM engine: groq (Llama4 Vision, best), clip, text")
    auto.add_argument("--threshold", type=float, default=0.85,
                      help="Min confidence to accept label (default: 0.85)")
    auto.add_argument("--limit", type=int, default=None)
    auto.add_argument("--all", action="store_true", dest="process_all",
                      help="Re-label all records, not just unlabeled ones")
    auto.set_defaults(func=cmd_auto_label)

    # ── Full Pipeline (One-shot end-to-end) ────────────────────
    full = subparsers.add_parser(
        "run-full-pipeline",
        help="🤖 Run the complete pipeline end-to-end: Crawl → Auto-Label → Enrich → Build Graphs → Train",
    )
    full.add_argument("--crawl-limit", type=int, default=50, help="Number of posts to crawl")
    full.add_argument("--images-only", action="store_true", help="Only crawl posts with images")
    full.add_argument("--subreddits", nargs="+", default=None)
    full.add_argument("--llm", choices=["groq", "clip", "text"], default="groq",
                      help="LLM engine for auto-labeling (default: groq)")
    full.add_argument("--confidence-threshold", type=float, default=0.85,
                      help="Min confidence for auto-labeling (default: 0.85)")
    full.add_argument("--require-human-review", action="store_true",
                      help="Export predictions to LS if confidence is below threshold")
    full.add_argument("--enrich-delay", type=float, default=2.0,
                      help="Delay between Reddit API calls during enrichment")
    full.add_argument("--train-model", choices=["multimodal_gnn", "gnn"], default="multimodal_gnn")
    full.add_argument("--epochs", type=int, default=30)
    full.add_argument("--batch_size", type=int, default=32)
    full.set_defaults(func=cmd_full_pipeline)

    # ── Train All Models ────────────────────
    train_all = subparsers.add_parser("train-all", help="🏆 Train all models sequentially")
    train_all.add_argument("--epochs", type=int, default=20)
    train_all.add_argument("--batch_size", type=int, default=16)
    train_all.set_defaults(func=cmd_train_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
