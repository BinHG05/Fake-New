"""
Graph Builder CLI for Multimodal Cascade GNN.
Constructs PyG cascade graphs from enriched JSONL records.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fakenews.config.settings import load_config
from src.fakenews.graphs.cascade import CascadeGraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("BuildGraphs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cascade Graphs from JSONL")
    parser.add_argument("--config", type=str, default="configs/canonical.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})

    input_file = Path(args.input or ds_cfg.get("metadata", "data/all_merged_binary.jsonl"))
    if not input_file.is_absolute():
        input_file = (PROJECT_ROOT / input_file).resolve()

    out_dir = Path(args.output_dir or ds_cfg.get("graph_dir", "data/processed_graphs_v3"))
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading input records from %s", input_file)
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit is not None and args.limit > 0:
        records = records[: args.limit]

    logger.info("Building graphs for %d records -> %s", len(records), out_dir)
    builder = CascadeGraphBuilder()
    built = 0
    skipped = 0

    for item in tqdm(records, desc="Building cascade graphs"):
        post_id = str(item.get("id", "")).strip()
        if not post_id:
            continue
        target_path = out_dir / f"{post_id}.pt"
        if target_path.exists() and not args.force:
            skipped += 1
            continue

        graph = builder.build_graph(item)
        torch.save(graph, target_path)
        built += 1

    logger.info("Graph generation finished: Built=%d, Skipped=%d", built, skipped)


if __name__ == "__main__":
    main()

