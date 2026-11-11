"""
Verification script for Fake News Demo Pipeline.
Tests the FakeNewsPredictor class with various scenarios.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Disable verbose logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.deployment.predictor import FakeNewsPredictor, PredictorError

def test_health():
    print("\n=== [1] Testing Predictor Health ===")
    try:
        predictor = FakeNewsPredictor()
        health = predictor.health()
        print(f"Predictor Status: {'READY' if health['ready'] else 'NOT READY'}")
        print(f"Model Type: {health['model_type']}")
        print(f"Checkpoint: {health['checkpoint']}")
        print(f"Device: {health['device']}")
        print(f"Classes: {health.get('classes', 'N/A')}")
        return predictor
    except Exception as e:
        print(f"[!] Health check failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_url_inference(predictor):
    print("\n=== [2] Testing Single URL Inference ===")
    # A known stable Reddit post
    test_url = "https://www.reddit.com/r/technology/comments/1bgsv8r/the_legal_battle_over_openai_and_copyright/"
    
    try:
        print(f"Predicting URL: {test_url}")
        result = predictor.predict_url(test_url)
        
        print(f"[+] Success!")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Probabilities: {result['probabilities']}")
        print(f"   Title: {result['post_info']['title'][:50]}...")
        print(f"   Nodes: {result['graph_info']['num_nodes']}, Edges: {result['graph_info']['num_edges']}")
        if result.get('warnings'):
            print(f"   Warnings: {result['warnings']}")
            
    except PredictorError as e:
        print(f"[!] Predictor Error: {e}")
    except Exception as e:
        print(f"[!] Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

def test_batch_inference(predictor):
    print("\n=== [3] Testing Batch Inference ===")
    dummy_file = PROJECT_ROOT / "data" / "test_demo_batch.jsonl"
    
    dummy_data = [
        {
            "id": "test_1",
            "title": "Genuine news about technology",
            "raw_text": "Genuine news about technology breakthrough in AI.",
            "label": "REAL",
            "timestamp": 1700000000,
            "cascade": []
        },
        {
            "id": "test_2",
            "title": "Breaking: Aliens landed in Central Park",
            "raw_text": "Aliens have landed in Central Park and are eating hotdogs.",
            "label": "FAKE",
            "timestamp": 1700000001,
            "cascade": []
        }
    ]
    
    try:
        with open(dummy_file, "w", encoding="utf-8") as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Predicting batch from: {dummy_file}")
        result = predictor.predict_batch(dummy_file)
        
        summary = result['summary']
        print(f"[+] Success!")
        print(f"   Total: {result['total']}")
        print(f"   Summary: REAL={summary['real_count']}, FAKE={summary['fake_count']}")
        print(f"   Accuracy: {summary['accuracy']}")
        print(f"   Avg Confidence: {summary['avg_confidence']}")
        
    except Exception as e:
        print(f"[!] Batch test failed: {e}")
    finally:
        if dummy_file.exists():
            dummy_file.unlink()

def main():
    print("Starting Demo Pipeline Verification...")
    
    predictor = test_health()
    if not predictor:
        print("Terminating verification due to initialization failure.")
        return
        
    test_url_inference(predictor)
    test_batch_inference(predictor)
    
    print("\n=== Verification Finished ===")

if __name__ == "__main__":
    main()
