"""
Unit tests for data contract validator.
"""

import json
import tempfile
from pathlib import Path
import pytest
from src.fakenews.labeling.validator import canonicalize_record, validate_dataset, ValidationError


def test_canonicalize_record_valid():
    raw = {
        "id": "post_123",
        "source": "reddit",
        "raw_text": "Scientists discover water on Mars.",
        "label_binary": "REAL",
        "split": "train",
        "image": "data/mars.jpg",
        "cascade": [{"id": "c1", "parent_id": "post_123", "text": "Amazing discovery!"}],
    }
    canonical = canonicalize_record(raw)
    assert canonical["id"] == "post_123"
    assert canonical["label_binary"] == "REAL"
    assert canonical["split"] == "train"
    assert canonical["media"]["available"] is True
    assert len(canonical["cascade"]) == 1


def test_canonicalize_record_missing_id():
    raw = {"raw_text": "Sample without ID", "label_binary": "REAL"}
    with pytest.raises(ValidationError):
        canonicalize_record(raw)


def test_validate_fixtures_passes():
    fixtures_file = "tests/fixtures/sample_posts.jsonl"
    report = validate_dataset(fixtures_file, raise_on_error=True)
    assert report["status"] == "PASS"
    assert report["total_records"] > 0
    assert report["unique_ids"] == report["total_records"]


def test_invalid_record_is_written_to_quarantine():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        source = temp_path / "invalid.jsonl"
        source.write_text('{"raw_text": "missing identifier", "label_binary": "REAL"}\n', encoding="utf-8")
        quarantine = temp_path / "quarantine.jsonl"

        report = validate_dataset(source, raise_on_error=False, quarantine_path=quarantine)

        assert report["status"] == "FAIL"
        assert report["quarantined_records"] == 1
        line = json.loads(quarantine.read_text(encoding="utf-8"))
        assert line["error_code"] == "MISSING_ID"

