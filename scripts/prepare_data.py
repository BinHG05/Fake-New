"""
Data Preparation & Validation CLI.
Validates the canonical dataset against schema rules and produces manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fakenews.config.settings import load_config
from src.fakenews.labeling.validator import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("PrepareData")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and Validate Dataset")
    parser.add_argument("--config", type=str, default="configs/canonical.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--graphs", type=str, default=None)
    parser.add_argument("--output_report", type=str, default="data/manifests/dataset_validation_report.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})

    input_file = Path(args.input or ds_cfg.get("metadata", "data/all_merged_binary.jsonl"))
    if not input_file.is_absolute():
        input_file = (PROJECT_ROOT / input_file).resolve()

    graph_dir = Path(args.graphs or ds_cfg.get("graph_dir", "data/processed_graphs_v3"))
    if not graph_dir.is_absolute():
        graph_dir = (PROJECT_ROOT / graph_dir).resolve()

    logger.info("Validating dataset: %s", input_file)
    logger.info("Checking graphs in: %s", graph_dir)

    report = validate_dataset(input_file, graph_dir=graph_dir, raise_on_error=False)

    out_rep = Path(args.output_report)
    if not out_rep.is_absolute():
        out_rep = (PROJECT_ROOT / out_rep).resolve()
    out_rep.parent.mkdir(parents=True, exist_ok=True)

    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation Status: %s (Errors: %d)", report["status"], report["error_count"])
    logger.info("Total records: %d | Unique: %d", report["total_records"], report["unique_ids"])
    logger.info("Labels: %s", report["label_distribution"])
    logger.info("Splits: %s", report["split_distribution"])
    logger.info("Sources: %s", report["source_distribution"])
    logger.info("Report written to %s", out_rep)


if __name__ == "__main__":
    main()

