"""
Integration tests for end-to-end pipeline components.
"""

from pathlib import Path
import pytest
from src.fakenews.config.settings import load_config
from src.fakenews.serving.predictor import FakeNewsPredictor


def test_predictor_integration():
    config = load_config("configs/canonical.yaml")
    predictor = FakeNewsPredictor(config_path="configs/canonical.yaml")

    sample_post = {
        "id": "integration_test_item",
        "clean_text": "Scientists develop breakthrough battery technology.",
        "image": None,
        "cascade": [],
    }

    result = predictor.predict_record(sample_post)
    assert result["id"] == "integration_test_item"
    assert result["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert "probability_fake" in result
    assert "probability_real" in result
    assert "missing_image" in result["warnings"]
    assert "missing_cascade_comments" in result["warnings"]

