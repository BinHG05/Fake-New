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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
