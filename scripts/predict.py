"""
Prediction CLI Entrypoint for Multimodal Cascade Fake News Pipeline.

Usage:
  python scripts/predict.py --input demo/sample_batch.jsonl
  python scripts/predict.py --text "Scientists discover chocolate cures aging"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fakenews.config.settings import load_config
from src.fakenews.serving.predictor import FakeNewsPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("PredictCLI")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Fake News with Multimodal Cascade GNN")
    parser.add_argument("--config", type=str, default="configs/canonical.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Path to batch JSONL file")
    parser.add_argument("--text", type=str, default=None, help="Single post text to classify")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for single post")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    predictor = FakeNewsPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    if args.input:
        res = predictor.predict_batch(args.input)
        print(f"\nBatch Prediction Summary (Total: {res['total_records']}):")
        print(f"  REAL:      {res['prediction_counts']['REAL']}")
        print(f"  FAKE:      {res['prediction_counts']['FAKE']}")
        print(f"  UNCERTAIN: {res['prediction_counts']['UNCERTAIN']}")
        print(f"  Avg Conf:  {res['average_confidence']:.4f}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Results saved to {args.output}")
    elif args.text:
        record = {
            "id": "single_cli_sample",
            "clean_text": args.text,
            "raw_text": args.text,
            "image": args.image,
            "cascade": [],
        }
        res = predictor.predict_record(record)
        print("\nSingle Prediction Result:")
        print(json.dumps(res, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
    else:
        print("Please provide --input <file.jsonl> or --text <raw_text>")


if __name__ == "__main__":
    main()

