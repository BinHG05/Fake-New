"""
Reproducible Smoke Test for Fake News Pipeline.
Validates end-to-end functionality from clean checkout:
1. Load sample dataset
2. Validate schema & contract
3. Build a multimodal cascade graph
4. Load canonical model checkpoint
5. Execute single and batch inference
6. Validate output JSON structure
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.fakenews.config.settings import load_config
from src.fakenews.graphs.cascade import CascadeGraphBuilder
from src.fakenews.labeling.validator import validate_dataset
from src.fakenews.models.multimodal_gnn import MultimodalCascadeGNN
from src.fakenews.serving.predictor import FakeNewsPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("SmokeTest")


def run_smoke_test() -> bool:
    logger.info("=" * 60)
    logger.info("STARTING PIPELINE SMOKE TEST")
    logger.info("=" * 60)

    # 1. Check Configuration
    cfg = load_config("configs/canonical.yaml")
    assert "dataset" in cfg, "Missing dataset config"
    assert "model" in cfg, "Missing model config"
    logger.info("  [1/6] Config loaded successfully.")

    # 2. Validate Sample Data Schema
    sample_file = PROJECT_ROOT / "tests" / "fixtures" / "sample_posts.jsonl"
    assert sample_file.exists(), f"Sample fixtures missing: {sample_file}"
    rep = validate_dataset(sample_file, raise_on_error=True)
    assert rep["status"] == "PASS", f"Validation status: {rep['status']}"
    assert rep["total_records"] > 0, "No records in sample file"
    logger.info("  [2/6] Sample dataset validated: %d records, 0 errors.", rep["total_records"])

    # 3. Build Cascade Graph
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_record = json.loads(next(f))

    builder = CascadeGraphBuilder()
    graph = builder.build_graph(sample_record)
    assert graph.x.size(1) == 1793, f"Expected node dim 1793, got {graph.x.size(1)}"
    assert graph.num_nodes >= 1, "Graph has no nodes"
    logger.info(
        "  [3/6] Cascade graph built successfully (Nodes=%d, Edges=%d, Dim=%d).",
        graph.num_nodes,
        graph.edge_index.size(1),
        graph.x.size(1),
    )

    # 4. Check Checkpoint Loading
    ckpt_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "multimodal_cascade_gnn_binary.pt"
    assert ckpt_path.exists(), f"Canonical checkpoint missing: {ckpt_path}"
    model = MultimodalCascadeGNN(
        input_dim=1793,
        text_dim=768,
        image_dim=1025,
        fusion_dim=512,
        hidden_dim=512,
        num_classes=2,
        num_gnn_layers=3,
        dropout=0.0,
        gnn_type="gat",
        fusion_type="cross_attention",
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    logger.info("  [4/6] Canonical model checkpoint loaded successfully (%s).", ckpt_path.name)

    # 5. Execute Inference via Predictor
    predictor = FakeNewsPredictor(checkpoint_path=ckpt_path)
    res_single = predictor.predict_record(sample_record)
    assert "prediction" in res_single, "Prediction field missing"
    assert res_single["prediction"] in {"REAL", "FAKE", "UNCERTAIN"}, "Invalid prediction label"
    assert "probability_fake" in res_single, "Probability field missing"
    assert 0.0 <= res_single["probability_fake"] <= 1.0, "Probability out of range"
    logger.info("  [5/6] Single inference output: %s", json.dumps(res_single))

    # 6. Execute Batch Prediction
    res_batch = predictor.predict_batch(sample_file)
    assert res_batch["total_records"] == rep["total_records"], "Mismatch in batch count"
    assert "average_confidence" in res_batch, "Missing average confidence"
    logger.info(
        "  [6/6] Batch inference executed: %d records processed (Avg Conf: %.4f).",
        res_batch["total_records"],
        res_batch["average_confidence"],
    )

    logger.info("=" * 60)
    logger.info("SMOKE TEST RESULT: ALL CHECKS PASSED (100% REPRODUCIBLE)")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    except Exception as exc:
        logger.exception("SMOKE TEST FAILED: %s", exc)
        sys.exit(1)

