"""
Demo Verification Script for FakeNews Detection Pipeline.
Validates the demo app API endpoints:
- Checkpoint loading
- Real probability output
- Missing image warnings
- Official canonical metrics & confusion matrix
- Batch inference
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("VerifyDemo")


def verify_demo() -> bool:
    logger.info("=" * 60)
    logger.info("STARTING DEMO APP VERIFICATION")
    logger.info("=" * 60)

    from demo.app import app

    client = app.test_client()

    # 1. Test Health endpoint
    logger.info("Testing GET /api/health ...")
    r_health = client.get("/api/health")
    assert r_health.status_code == 200, f"Health status {r_health.status_code}"
    health_data = r_health.get_json()
    assert health_data["status"] == "HEALTHY", f"Unhealthy status: {health_data}"
    assert "multimodal_cascade_gnn_binary" in health_data["checkpoint"], f"Unexpected checkpoint: {health_data['checkpoint']}"
    logger.info("  [+] Health Check Passed: %s", health_data["checkpoint"])

    # 2. Test Official Metrics endpoint
    logger.info("Testing GET /api/metrics ...")
    r_metrics = client.get("/api/metrics")
    assert r_metrics.status_code == 200, f"Metrics status {r_metrics.status_code}"
    metrics_data = r_metrics.get_json()
    assert "accuracy" in metrics_data, "Missing accuracy in metrics"
    assert "f1" in metrics_data, "Missing F1 in metrics"
    assert "confusion_matrix" in metrics_data, "Missing confusion matrix"
    logger.info("  [+] Metrics Check Passed: Accuracy=%.4f, F1=%.4f", metrics_data["accuracy"], metrics_data["f1"])

    # 3. Test Single Prediction with Missing Image Warning
    logger.info("Testing POST /api/predict/record (Text-only payload) ...")
    r_rec = client.post(
        "/api/predict/record",
        json={"clean_text": "Alien spaceships spotted hovering over Tokyo, official news confirms."},
    )
    assert r_rec.status_code == 200, f"Record predict status {r_rec.status_code}"
    rec_data = r_rec.get_json()
    assert "prediction" in rec_data, "Missing prediction"
    assert "probability_fake" in rec_data, "Missing probability_fake"
    assert "missing_image" in rec_data["warnings"], "Expected missing_image warning"
    logger.info(
        "  [+] Single Record Prediction Passed: Pred=%s, P(Fake)=%.4f, Warnings=%s",
        rec_data["prediction"],
        rec_data["probability_fake"],
        rec_data["warnings"],
    )

    # 4. Test Batch Prediction
    logger.info("Testing POST /api/predict/batch ...")
    r_batch = client.post("/api/predict/batch")
    assert r_batch.status_code == 200, f"Batch predict status {r_batch.status_code}"
    batch_data = r_batch.get_json()
    assert batch_data["total_records"] > 0, "No records returned in batch"
    logger.info("  [+] Batch Prediction Passed: Total=%d, Counts=%s", batch_data["total_records"], batch_data["prediction_counts"])

    # 5. Test Confusion Matrix Figure
    logger.info("Testing GET /api/figures/confusion_matrix ...")
    r_fig = client.get("/api/figures/confusion_matrix")
    assert r_fig.status_code == 200, f"Figure status {r_fig.status_code}"
    assert r_fig.mimetype == "image/png", f"Invalid mimetype {r_fig.mimetype}"
    logger.info("  [+] Confusion Matrix PNG Served Successfully (%d bytes)", len(r_fig.data))

    logger.info("=" * 60)
    logger.info("DEMO APP VERIFICATION: ALL 5 CHECKS PASSED")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        ok = verify_demo()
        sys.exit(0 if ok else 1)
    except Exception as exc:
        logger.exception("DEMO VERIFICATION FAILED: %s", exc)
        sys.exit(1)
