"""CLI entrypoint for the Bronze-to-Silver PySpark ETL job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fakenews.config.settings import load_config
from src.fakenews.preprocessing.pyspark_etl import run_bronze_to_silver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical Bronze-to-Silver PySpark ETL")
    parser.add_argument("--config", default="configs/canonical.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--parquet-output", default="data/silver/canonical_posts")
    parser.add_argument("--jsonl-output", default="data/silver/canonical_records.jsonl")
    parser.add_argument("--quarantine-output", default="data/quarantine/spark_rejected_records")
    parser.add_argument("--manifest-output", default="data/manifests/pyspark_etl_manifest.json")
    parser.add_argument("--partitions", type=int, default=24)
    parser.add_argument("--master", default=None, help="Spark master, e.g. local[*]")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or config.get("dataset", {}).get("metadata", "data/all_merged_binary.jsonl")
    manifest = run_bronze_to_silver(
        input_path=PROJECT_ROOT / input_path,
        parquet_output=PROJECT_ROOT / args.parquet_output,
        jsonl_output=PROJECT_ROOT / args.jsonl_output,
        quarantine_output=PROJECT_ROOT / args.quarantine_output,
        manifest_output=PROJECT_ROOT / args.manifest_output,
        target_partitions=args.partitions,
        master=args.master,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
