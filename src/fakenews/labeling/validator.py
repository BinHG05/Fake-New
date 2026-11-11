"""
Data contract validator for fake news pipeline.
Validates schema conformance, label policy, duplicates, leakage, and graph consistency.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from src.fakenews.labeling.labels import BINARY_LABEL_MAP, derive_binary_label

logger = logging.getLogger(__name__)

CANONICAL_SPLITS = {"train", "val", "test"}


class ValidationError(Exception):
    """Raised when data contract validation fails."""


def canonicalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Transforms any raw or merged JSONL record into the official canonical contract:
    {
      "id": "...",
      "source": "reddit",
      "raw_text": "...",
      "clean_text": "...",
      "label_binary": "REAL",
      "split": "train",
      "media": {
        "url": "...",
        "processed_path": "...",
        "available": bool
      },
      "cascade": [...],
      "metadata": {...}
    }
    """
    post_id = str(raw.get("id", "")).strip()
    if not post_id:
        raise ValidationError("Record missing required field 'id'")

    label_bin, _ = derive_binary_label(raw)
    split = str(raw.get("split", "train")).strip().lower()
    if split not in CANONICAL_SPLITS:
        raise ValidationError(f"Invalid split '{split}' for record {post_id}")

    raw_text = raw.get("raw_text") or raw.get("clean_text") or raw.get("title") or ""
    clean_text = raw.get("clean_text") or raw_text

    # Media mapping
    img_info = raw.get("image_info") if isinstance(raw.get("image_info"), dict) else {}
    media_url = raw.get("image_url") or raw.get("media_url") or img_info.get("url") or ""
    media_path = raw.get("image") or raw.get("image_path") or img_info.get("processed_path") or ""
    media_available = bool(media_path or media_url)

    media = {
        "url": str(media_url or ""),
        "processed_path": str(media_path or ""),
        "available": media_available,
    }

    cascade = raw.get("cascade")
    if not isinstance(cascade, list):
        cascade = []

    source = str(raw.get("source_dataset") or raw.get("source") or "reddit").strip().lower()

    metadata = {
        "original_label": raw.get("original_label"),
        "user_id": raw.get("user_id"),
        "timestamp": raw.get("timestamp"),
    }
    if isinstance(raw.get("metadata"), dict):
        metadata.update(raw["metadata"])

    return {
        "id": post_id,
        "source": source,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "label_binary": label_bin,
        "split": split,
        "media": media,
        "cascade": cascade,
        "metadata": metadata,
    }


def validate_dataset(
    jsonl_path: str | Path,
    graph_dir: str | Path | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """
    Validates a dataset JSONL file against the canonical data contract.
    Returns statistical report.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    total_records = 0
    seen_ids: set[str] = set()
    split_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    labels_counter = Counter()
    sources_counter = Counter()
    split_counter = Counter()
    missing_media_count = 0
    errors: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error: {e}")
                continue

            # ID validation
            rec_id = str(raw_item.get("id", "")).strip()
            if not rec_id:
                errors.append(f"Line {line_num}: missing ID")
                continue
            if rec_id in seen_ids:
                errors.append(f"Line {line_num}: duplicate ID '{rec_id}'")
            seen_ids.add(rec_id)

            # Label validation
            try:
                label_bin, _ = derive_binary_label(raw_item)
                labels_counter[label_bin] += 1
            except ValueError as e:
                errors.append(f"Line {line_num} (id={rec_id}): invalid label - {e}")
                continue

            # Split validation
            split = str(raw_item.get("split", "")).strip().lower()
            if split not in CANONICAL_SPLITS:
                errors.append(f"Line {line_num} (id={rec_id}): invalid split '{split}'")
            else:
                split_counter[split] += 1
                split_ids[split].add(rec_id)

            # Source validation
            source = str(raw_item.get("source_dataset") or raw_item.get("source") or "unknown").strip().lower()
            sources_counter[source] += 1

            # Media check
            media = raw_item.get("media")
            if isinstance(media, dict):
                if not media.get("available") and not media.get("processed_path"):
                    missing_media_count += 1
            else:
                img = raw_item.get("image") or raw_item.get("image_path")
                if not img:
                    missing_media_count += 1

    # Check for split leakage
    train_val_overlap = split_ids["train"].intersection(split_ids["val"])
    train_test_overlap = split_ids["train"].intersection(split_ids["test"])
    val_test_overlap = split_ids["val"].intersection(split_ids["test"])

    if train_val_overlap:
        errors.append(f"Split leakage: {len(train_val_overlap)} IDs appear in both train and val")
    if train_test_overlap:
        errors.append(f"Split leakage: {len(train_test_overlap)} IDs appear in both train and test")
    if val_test_overlap:
        errors.append(f"Split leakage: {len(val_test_overlap)} IDs appear in both val and test")

    # Graph alignment check if graph_dir provided
    graph_check: dict[str, Any] = {"checked": False}
    if graph_dir is not None:
        gpath = Path(graph_dir)
        if gpath.exists():
            graph_files = {p.stem for p in gpath.glob("*.pt")}
            matched_graphs = seen_ids.intersection(graph_files)
            missing_graphs = seen_ids - graph_files
            extra_graphs = graph_files - seen_ids
            graph_check = {
                "checked": True,
                "total_graphs_on_disk": len(graph_files),
                "matched_records": len(matched_graphs),
                "missing_graphs_for_records": len(missing_graphs),
                "extra_graphs_without_record": len(extra_graphs),
            }
            if len(missing_graphs) > 0:
                logger.warning(
                    f"{len(missing_graphs)} dataset records have no matching graph file in {graph_dir}"
                )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "total_records": total_records,
        "unique_ids": len(seen_ids),
        "label_distribution": dict(labels_counter),
        "split_distribution": dict(split_counter),
        "source_distribution": dict(sources_counter),
        "missing_media_count": missing_media_count,
        "split_leakage": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
        },
        "graph_check": graph_check,
        "error_count": len(errors),
        "errors": errors[:50],  # cap preview
    }

    if raise_on_error and errors:
        raise ValidationError(
            f"Dataset validation failed with {len(errors)} errors:\n"
            + "\n".join(errors[:10])
        )

    return report

