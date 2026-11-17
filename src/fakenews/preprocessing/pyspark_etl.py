"""PySpark Bronze-to-Silver ETL for multimodal fake-news records.

This module is intentionally separate from model code. It demonstrates the
scalable data-engineering path used to standardize JSONL, isolate rejected
records, rebalance partitions, and materialize query-efficient Parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_pyspark():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql import types as T
    except ImportError as exc:  # keeps local unit tests lightweight
        raise RuntimeError(
            "PySpark is required for Bronze-to-Silver ETL. Install with "
            "`pip install -e .[spark]` or use environment.yml."
        ) from exc
    return SparkSession, F, T


def build_spark_session(app_name: str = "fakenews-bronze-to-silver", master: str | None = None):
    """Create a Spark session with conservative local defaults."""
    SparkSession, _, _ = _require_pyspark()
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))
        .config("spark.sql.parquet.compression.codec", "snappy")
    )
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def run_bronze_to_silver(
    input_path: str | Path,
    parquet_output: str | Path,
    jsonl_output: str | Path,
    quarantine_output: str | Path,
    manifest_output: str | Path,
    target_partitions: int = 24,
    master: str | None = None,
) -> dict[str, Any]:
    """Normalize canonical JSONL, quarantine invalid rows, and write Silver data.

    The job uses a broadcast join between post-level metadata and exploded
    cascade comments. This avoids shuffling the small post table when enriching
    high-cardinality comment records. Final Silver storage is partitioned by
    source and event_date with Snappy Parquet; a JSONL compatibility export is
    also emitted for the existing PyG graph builder.
    """
    SparkSession, F, T = _require_pyspark()
    spark = build_spark_session(master=master)
    input_path = str(Path(input_path).resolve())
    parquet_output = str(Path(parquet_output).resolve())
    jsonl_output = str(Path(jsonl_output).resolve())
    quarantine_output = str(Path(quarantine_output).resolve())
    manifest_output = Path(manifest_output).resolve()

    # permissive ingest preserves malformed rows for quarantine rather than
    # failing an entire batch at the first bad JSON record.
    bronze = (
        spark.read.option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .json(input_path)
    )
    if "_corrupt_record" not in bronze.columns:
        bronze = bronze.withColumn("_corrupt_record", F.lit(None).cast("string"))

    # The canonical export is the contract, but the checked-in source dataset
    # still contains legacy names such as ``source_dataset`` and ``label``.
    # Resolve those aliases before applying the Silver schema.
    available = set(bronze.columns)
    source_expr = F.col("source") if "source" in available else F.lit(None)
    source_dataset_expr = F.col("source_dataset") if "source_dataset" in available else F.lit(None)
    label_expr = F.col("label_binary") if "label_binary" in available else F.col("label")
    raw_text_expr = F.col("raw_text") if "raw_text" in available else F.col("text")
    clean_text_expr = F.col("clean_text") if "clean_text" in available else raw_text_expr
    timestamp_expr = F.col("timestamp") if "timestamp" in available else F.lit(None)
    if "metadata" in available and hasattr(bronze.schema["metadata"].dataType, "fieldNames"):
        if "timestamp" in bronze.schema["metadata"].dataType.fieldNames():
            timestamp_expr = F.coalesce(F.col("metadata.timestamp"), timestamp_expr)

    required_columns = {"id", "split"}
    missing_columns = required_columns - set(bronze.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required canonical columns: {sorted(missing_columns)}")

    normalized = (
        bronze
        .withColumn("id", F.trim(F.col("id").cast("string")))
        .withColumn("source", F.lower(F.trim(F.coalesce(source_expr, source_dataset_expr, F.lit("unknown")))))
        .withColumn("label_binary", F.upper(F.trim(label_expr.cast("string"))))
        .withColumn("split", F.lower(F.trim(F.col("split").cast("string"))))
        .withColumn("raw_text", F.trim(raw_text_expr.cast("string")))
        .withColumn("clean_text", F.trim(clean_text_expr.cast("string")))
        .withColumn(
            "event_date",
            F.to_date(F.from_unixtime(timestamp_expr.cast("bigint"))),
        )
    )

    invalid_reason = (
        F.when(F.col("_corrupt_record").isNotNull(), F.lit("INVALID_JSON"))
        .when(F.col("id").isNull() | (F.col("id") == ""), F.lit("MISSING_ID"))
        .when(~F.col("label_binary").isin("REAL", "FAKE"), F.lit("INVALID_LABEL"))
        .when(~F.col("split").isin("train", "val", "test"), F.lit("INVALID_SPLIT"))
        .when(F.col("source").isin("", "unknown"), F.lit("MISSING_SOURCE"))
        .when(F.col("raw_text").isNull() | (F.col("raw_text") == ""), F.lit("MISSING_TEXT"))
    )
    staged = normalized.withColumn("quarantine_reason", invalid_reason)
    quarantined = staged.filter(F.col("quarantine_reason").isNotNull())
    valid = staged.filter(F.col("quarantine_reason").isNull()).drop("_corrupt_record", "quarantine_reason")

    # Deterministic de-duplication: retain the latest timestamp per post ID.
    from pyspark.sql.window import Window
    dedupe_window = Window.partitionBy("id").orderBy(F.col("timestamp").desc_nulls_last())
    valid = valid.withColumn("_rank", F.row_number().over(dedupe_window)).filter(F.col("_rank") == 1).drop("_rank")

    # Demonstrate a real broadcast join: enrich exploded comments with the small
    # post metadata relation before reassembling aggregate cascade statistics.
    post_metadata = F.broadcast(valid.select("id", "source", "split", "label_binary"))
    comments = (
        valid.select("id", F.explode_outer("cascade").alias("comment"))
        .filter(F.col("comment").isNotNull())
        .select("id", F.col("comment.id").alias("comment_id"), F.col("comment.level").alias("comment_level"))
        .join(post_metadata, on="id", how="left")
    )
    comment_stats = comments.groupBy("id").agg(
        F.count("comment_id").alias("cascade_comment_count"),
        F.max("comment_level").alias("cascade_max_depth"),
    )
    silver = valid.join(comment_stats, on="id", how="left").fillna({"cascade_comment_count": 0, "cascade_max_depth": 0})

    # Rebalance before write and partition storage by stable query predicates.
    silver = silver.repartition(max(1, int(target_partitions)), "source", "event_date")
    silver.write.mode("overwrite").partitionBy("source", "event_date").parquet(parquet_output)
    quarantined.write.mode("overwrite").json(quarantine_output)

    # Existing graph builder consumes a single JSONL file, so retain a compact
    # compatibility export in addition to the analytics-oriented Parquet table.
    jsonl_target = Path(jsonl_output)
    jsonl_target.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.asDict(recursive=True) for row in silver.toLocalIterator()]
    with jsonl_target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    manifest = {
        "input_path": input_path,
        "parquet_output": parquet_output,
        "jsonl_output": str(jsonl_target),
        "quarantine_output": quarantine_output,
        "valid_records": len(rows),
        "quarantined_records": quarantined.count(),
        "target_partitions": int(target_partitions),
        "storage_format": "snappy parquet partitioned by source,event_date",
        "join_strategy": "broadcast(post_metadata) onto exploded cascade comments",
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    spark.stop()
    return manifest


__all__ = ["build_spark_session", "run_bronze_to_silver"]
