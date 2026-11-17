"""
Build per-post cascade graph files from an enriched JSONL dataset.

This is a resumable, CLI-friendly replacement for build_final_graphs.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.cascade_graph_builder import CascadeGraphBuilder


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_graphs(input_path: Path, output_dir: Path, force: bool = False, limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(input_path)
    builder = CascadeGraphBuilder()

    built = 0
    skipped = 0
    failed = 0

    for index, item in enumerate(tqdm(records, desc="Building graphs"), start=1):
        if limit is not None and built >= limit:
            break

        post_id = str(item.get("id", "")).strip()
        if not post_id:
            failed += 1
            continue

        save_path = output_dir / f"{post_id}.pt"
        if save_path.exists() and not force:
            skipped += 1
            continue

        try:
            data = builder.build_graph(item)
            if data is None:
                failed += 1
                continue
            torch.save(data, save_path)
            built += 1
        except Exception as exc:
            failed += 1
            print(f"Failed to build graph for {post_id}: {exc}")

    print("=" * 60)
    print("GRAPH BUILD COMPLETE")
    print(f"Input records: {len(records)}")
    print(f"New graphs built: {built}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output folder: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cascade graphs from enriched JSONL")
    parser.add_argument(
        "--input",
        default="data/reddit_enriched_data.jsonl",
        help="Input enriched JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/processed_graphs",
        help="Output folder for .pt graphs",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if .pt file already exists")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of new graphs to build")
    args = parser.parse_args()

    build_graphs(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
