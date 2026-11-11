"""
Unit tests for data contract validator.
"""

import json
import pytest
from src.fakenews.labeling.validator import canonicalize_record, validate_dataset, ValidationError


def test_canonicalize_record_valid():
    raw = {
        "id": "post_123",
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
