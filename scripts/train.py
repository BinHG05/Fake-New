"""
Official Training Entrypoint for FakeNews Multimodal Cascade GNN.

Usage:
  python scripts/train.py --config configs/canonical.yaml
  python scripts/train.py --config configs/canonical.yaml --epochs 1 --limit 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fakenews.config.settings import load_config
from src.fakenews.training.trainer import train_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("TrainEntrypoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multimodal Cascade GNN Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/canonical.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of dataset records (for smoke tests and debugging)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    args = parser.parse_args()

    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)

    if args.limit is not None and args.limit > 0:
        # Protect canonical checkpoint from debug/smoke limit runs
        if config.get("training", {}).get("checkpoint_name") == "multimodal_cascade_gnn_binary.pt":
            config.setdefault("training", {})["checkpoint_name"] = "dev_multimodal_cascade_gnn.pt"

    result = train_pipeline(
        config=config,
        epochs_override=args.epochs,
        limit_override=args.limit,
    )

    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully.")
    logger.info("Checkpoint saved to: %s", result["checkpoint_path"])
    if result.get("test_metrics"):
        logger.info(
            "Test Accuracy: %.4f | Test F1: %.4f",
            result["test_metrics"]["accuracy"],
            result["test_metrics"]["f1"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

