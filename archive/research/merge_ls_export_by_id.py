"""
Merge a Label Studio JSON export into the project master JSONL by post id.

Duplicate ids are excluded from merge instead of overwriting existing records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_label_studio_export(input_path: Path) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    records = []
    for task in tasks:
        record = task.get("data", task).copy()

        if "data" not in task:
            for key in ["annotations", "predictions", "id", "created_at", "updated_at", "project"]:
                record.pop(key, None)

        annotations = task.get("annotations", [])
        label = None
        notes = None
        if annotations:
            last_annotation = annotations[-1]
            for result_item in last_annotation.get("result", []):
                from_name = result_item.get("from_name")
                if from_name in {"label", "label_binary", "label_fine", "choice"}:
                    choices = result_item.get("value", {}).get("choices", [])
                    if choices:
                        label = choices[0]
                if from_name == "notes":
                    notes = result_item.get("value", {}).get("text", [None])[0]

        if label:
            record["label"] = label
            record["manual_label"] = True
        else:
            record["manual_label"] = False

        if notes:
            record["notes"] = notes

        record["ls_id"] = task.get("id")
        records.append(record)

    return records


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def merge_by_id(existing_records: list[dict], new_records: list[dict]) -> tuple[list[dict], dict]:
    existing_ids = set()
    for record in existing_records:
        record_id = str(record.get("id", "")).strip()
        if record_id:
            existing_ids.add(record_id)

    accepted_new_records = []
    duplicate_in_export_ids = set()
    duplicate_with_master_ids = set()
    missing_id_count = 0
    seen_new_ids = set()

    for record in new_records:
        record_id = str(record.get("id", "")).strip()
        if not record_id:
            missing_id_count += 1
            continue
        if record_id in existing_ids:
            duplicate_with_master_ids.add(record_id)
            continue
        if record_id in seen_new_ids:
            duplicate_in_export_ids.add(record_id)
            continue

        seen_new_ids.add(record_id)
        accepted_new_records.append(record)

    merged = existing_records + accepted_new_records
    stats = {
        "accepted_new": len(accepted_new_records),
        "duplicate_with_master": sorted(duplicate_with_master_ids),
        "duplicate_in_export": sorted(duplicate_in_export_ids),
        "missing_id_count": missing_id_count,
    }
    return merged, stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Label Studio export into master JSONL by id")
    parser.add_argument("--input", required=True, help="Path to Label Studio JSON export")
    parser.add_argument(
        "--output",
        default="data/03_clean/Fakeddit/labeled_master.jsonl",
        help="Path to master JSONL output",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    parsed_records = parse_label_studio_export(input_path)
    existing_records = read_jsonl(output_path)
    merged_records, stats = merge_by_id(existing_records, parsed_records)
    write_jsonl(output_path, merged_records)

    print(f"Imported tasks: {len(parsed_records)}")
    print(f"Existing master records: {len(existing_records)}")
    print(f"Accepted new records: {stats['accepted_new']}")
    print(f"Skipped duplicate ids already in master: {len(stats['duplicate_with_master'])}")
    print(f"Skipped duplicate ids inside export: {len(stats['duplicate_in_export'])}")
    print(f"Skipped records without id: {stats['missing_id_count']}")
    print(f"Final master records: {len(merged_records)}")
    print(f"Saved to: {output_path}")

    if stats["duplicate_with_master"]:
        print("Duplicate ids already in master:")
        for record_id in stats["duplicate_with_master"][:20]:
            print(f"  - {record_id}")
        if len(stats["duplicate_with_master"]) > 20:
            print(f"  ... and {len(stats['duplicate_with_master']) - 20} more")

    if stats["duplicate_in_export"]:
        print("Duplicate ids inside export:")
        for record_id in stats["duplicate_in_export"][:20]:
            print(f"  - {record_id}")
        if len(stats["duplicate_in_export"]) > 20:
            print(f"  ... and {len(stats['duplicate_in_export']) - 20} more")


if __name__ == "__main__":
    main()
