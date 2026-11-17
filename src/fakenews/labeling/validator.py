"""Contract validation, lineage reporting, and quarantine handling.

The Bronze-to-Silver boundary is deliberately strict: malformed source records
are emitted to quarantine with machine-readable failure reasons instead of
being silently coerced into the canonical dataset.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from src.fakenews.labeling.contracts import CanonicalPostContract
from src.fakenews.labeling.labels import derive_binary_label

logger = logging.getLogger(__name__)

CANONICAL_SPLITS = {"train", "val", "test"}


class ValidationError(Exception):
    """Raised when validation is configured to fail on invalid records."""


def _error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "missing required field 'id'" in message or ("id" in message and "required" in message):
        return "MISSING_ID"
    if "label" in message:
        return "INVALID_LABEL"
    if "split" in message:
        return "INVALID_SPLIT"
    if "json" in message:
        return "INVALID_JSON"
    if "cascade" in message or "comment" in message:
        return "INVALID_CASCADE"
    return "CONTRACT_VIOLATION"


def canonicalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a legacy record to the canonical shape, then validate it strictly."""
    post_id = str(raw.get("id", "")).strip()
    if not post_id:
        raise ValidationError("Record missing required field 'id'")

    label_bin, _ = derive_binary_label(raw)
    split = str(raw.get("split", "")).strip().lower()
    if split not in CANONICAL_SPLITS:
        raise ValidationError(f"Invalid split '{split}' for record {post_id}")

    raw_text = str(raw.get("raw_text") or raw.get("clean_text") or raw.get("title") or "").strip()
    clean_text = str(raw.get("clean_text") or raw_text).strip()
    if not raw_text or not clean_text:
        raise ValidationError(f"Record {post_id} is missing usable text")

    img_info = raw.get("image_info") if isinstance(raw.get("image_info"), dict) else {}
    existing_media = raw.get("media") if isinstance(raw.get("media"), dict) else {}
    media_url = existing_media.get("url") or raw.get("image_url") or raw.get("media_url") or img_info.get("url") or ""
    media_path = existing_media.get("processed_path") or raw.get("image") or raw.get("image_path") or img_info.get("processed_path") or ""
    media = {
        "url": str(media_url or ""),
        "processed_path": str(media_path or ""),
        "available": bool(existing_media.get("available") or media_path or media_url),
    }

    cascade = raw.get("cascade")
    if cascade is None:
        cascade = []
    if not isinstance(cascade, list):
        raise ValidationError(f"Record {post_id} has non-list cascade")

    source = str(raw.get("source_dataset") or raw.get("source") or "unknown").strip().lower()
    if source == "unknown":
        raise ValidationError(f"Record {post_id} is missing source lineage")

    metadata = {
        "original_label": raw.get("original_label"),
        "user_id": raw.get("user_id"),
        "timestamp": raw.get("timestamp"),
    }
    if isinstance(raw.get("metadata"), dict):
        metadata.update(raw["metadata"])

    candidate = {
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
    try:
        return CanonicalPostContract.model_validate(candidate).model_dump(mode="json")
    except PydanticValidationError as exc:
        raise ValidationError(str(exc)) from exc


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_dataset(
    jsonl_path: str | Path,
    graph_dir: str | Path | None = None,
    raise_on_error: bool = True,
    quarantine_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate JSONL and optionally persist rejected records to quarantine."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    total_records = valid_records = missing_media_count = 0
    seen_ids: set[str] = set()
    split_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    labels_counter: Counter[str] = Counter()
    sources_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    errors: list[str] = []
    rejected: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total_records += 1
            raw_item: dict[str, Any] | None = None
            try:
                raw_item = json.loads(line)
                if not isinstance(raw_item, dict):
                    raise ValidationError("JSON line must be an object")
                normalized = canonicalize_record(raw_item)
                rec_id = normalized["id"]
                if rec_id in seen_ids:
                    raise ValidationError(f"Duplicate ID '{rec_id}'")

                seen_ids.add(rec_id)
                valid_records += 1
                labels_counter[normalized["label_binary"]] += 1
                sources_counter[normalized["source"]] += 1
                split = normalized["split"]
                split_counter[split] += 1
                split_ids[split].add(rec_id)
                if not normalized["media"]["available"]:
                    missing_media_count += 1
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                code = _error_code(exc)
                errors.append(f"Line {line_num}: {code} - {exc}")
                rejected.append({
                    "source_file": str(path),
                    "line_number": line_num,
                    "error_code": code,
                    "error_message": str(exc),
                    "record": raw_item if raw_item is not None else {"raw_line": line[:4000]},
                })

    overlaps = {
        "train_val_overlap": len(split_ids["train"] & split_ids["val"]),
        "train_test_overlap": len(split_ids["train"] & split_ids["test"]),
        "val_test_overlap": len(split_ids["val"] & split_ids["test"]),
    }
    for name, count in overlaps.items():
        if count:
            errors.append(f"Split leakage: {name}={count}")

    graph_check: dict[str, Any] = {"checked": False}
    if graph_dir is not None:
        gpath = Path(graph_dir)
        if gpath.exists():
            graph_files = {p.stem for p in gpath.glob("*.pt")}
            missing_graphs = seen_ids - graph_files
            graph_check = {
                "checked": True,
                "total_graphs_on_disk": len(graph_files),
                "matched_records": len(seen_ids & graph_files),
                "missing_graphs_for_records": len(missing_graphs),
                "extra_graphs_without_record": len(graph_files - seen_ids),
            }
            if missing_graphs:
                errors.append(f"Graph alignment: {len(missing_graphs)} records have no graph")

    if quarantine_path is not None:
        _write_jsonl(Path(quarantine_path), rejected)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "total_records": total_records,
        "valid_records": valid_records,
        "quarantined_records": len(rejected),
        "unique_ids": len(seen_ids),
        "label_distribution": dict(labels_counter),
        "split_distribution": dict(split_counter),
        "source_distribution": dict(sources_counter),
        "missing_media_count": missing_media_count,
        "split_leakage": overlaps,
        "graph_check": graph_check,
        "error_count": len(errors),
        "errors": errors[:50],
    }
    if raise_on_error and errors:
        raise ValidationError(f"Dataset validation failed with {len(errors)} errors:\n" + "\n".join(errors[:10]))
    return report


__all__ = ["ValidationError", "canonicalize_record", "validate_dataset"]
