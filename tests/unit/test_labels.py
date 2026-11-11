"""
Unit tests for binary label derivation and normalization.
"""

import pytest
from src.fakenews.labeling.labels import (
    BINARY_LABEL_MAP,
    derive_binary_label,
    normalize_label_str,
)


def test_normalize_label_str():
    assert normalize_label_str("real") == "REAL"
    assert normalize_label_str(" Fake ") == "FAKE"
    assert normalize_label_str("pants-on-fire") == "PANTS_ON_FIRE"
    assert normalize_label_str(None) == ""


def test_derive_binary_label_direct():
    rec_real = {"id": "1", "label_binary": "REAL"}
    rec_fake = {"id": "2", "label_binary": "FAKE"}
    assert derive_binary_label(rec_real) == ("REAL", 0)
    assert derive_binary_label(rec_fake) == ("FAKE", 1)


def test_derive_binary_label_fakeddit():
    rec_true = {"id": "3", "source_dataset": "Fakeddit", "original_label": "True"}
    rec_fake = {"id": "4", "source_dataset": "Fakeddit", "original_label": "Fake"}
    assert derive_binary_label(rec_true) == ("REAL", 0)
    assert derive_binary_label(rec_fake) == ("FAKE", 1)


def test_derive_binary_label_reddit_6class():
    rec_mostly = {"id": "5", "source_dataset": "Reddit", "label": "mostly_true"}
    rec_pants = {"id": "6", "source_dataset": "Reddit", "label": "pants_on_fire"}
    assert derive_binary_label(rec_mostly) == ("REAL", 0)
    assert derive_binary_label(rec_pants) == ("FAKE", 1)


def test_derive_binary_label_invalid():
    rec_unlabeled = {"id": "7", "label": "Unlabeled"}
    with pytest.raises(ValueError):
        derive_binary_label(rec_unlabeled)

