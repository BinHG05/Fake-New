"""
Official Evaluation Entrypoint for FakeNews Multimodal Cascade GNN.

Usage:
  python scripts/evaluate.py --config configs/canonical.yaml --checkpoint artifacts/checkpoints/multimodal_cascade_gnn_binary.pt
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

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.fakenews.config.settings import load_config
from src.fakenews.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_reliability_curve,
)
from src.fakenews.models.multimodal_gnn import MultimodalCascadeGNN
from src.fakenews.training.dataset import load_cascade_graphs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("EvaluateEntrypoint")


def tune_threshold_on_val(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Finds optimal decision threshold on validation set."""
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for b in val_loader:
            b = b.to(device)
            logits = model(b.x, b.edge_index, b.batch)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(b.y.view(-1).cpu().numpy().tolist())

    best_thresh = 0.5
    best_acc = 0.0
    val_probs = np.array(all_probs)
    val_targets = np.array(all_targets)

    for t in np.arange(0.30, 0.71, 0.02):
        preds = (val_probs >= t).astype(int)
        acc = float((preds == val_targets).mean())
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(round(t, 2))

    return best_thresh, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Cascade GNN Checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/canonical.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/multimodal_cascade_gnn_binary.pt",
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--tune_threshold",
        action="store_true",
        help="Tune threshold on validation set prior to test evaluation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})
    m_cfg = config.get("model", {})
    e_cfg = config.get("evaluation", {})

    root = PROJECT_ROOT
    meta_path = (root / ds_cfg.get("metadata", "data/all_merged_binary.jsonl")).resolve()
    graph_dir = (root / ds_cfg.get("graph_dir", "data/processed_graphs_v3")).resolve()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (root / ckpt_path).resolve()

    if not ckpt_path.exists():
        # Check fallback in models/checkpoints/
        alt_ckpt = root / "models" / "checkpoints" / ckpt_path.name
        if alt_ckpt.exists():
            ckpt_path = alt_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Evaluating Checkpoint: %s", ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graphs
    train_g, val_g, test_g = load_cascade_graphs(graph_dir, meta_path)
    logger.info("Graphs loaded: Train=%d, Val=%d, Test=%d", len(train_g), len(val_g), len(test_g))

    split_map = {"train": train_g, "val": val_g, "test": test_g}
    target_graphs = split_map[args.split]

    loader = DataLoader(target_graphs, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_g, batch_size=32, shuffle=False) if val_g else None

    # Init model
    model = MultimodalCascadeGNN(
        input_dim=int(m_cfg.get("text_dim", 768)) + int(m_cfg.get("image_dim", 1025)),
        text_dim=int(m_cfg.get("text_dim", 768)),
        image_dim=int(m_cfg.get("image_dim", 1025)),
        fusion_dim=int(m_cfg.get("fusion_dim", 512)),
        hidden_dim=int(m_cfg.get("hidden_dim", 512)),
        num_classes=int(m_cfg.get("num_classes", 2)),
        num_gnn_layers=int(m_cfg.get("num_gnn_layers", 3)),
        dropout=0.0,
        gnn_type=str(m_cfg.get("gnn", "gat")),
        fusion_type=str(m_cfg.get("fusion", "cross_attention")),
        learn_missing_image_emb=bool(m_cfg.get("learn_missing_image_emb", False)),
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    threshold = float(e_cfg.get("threshold", 0.5))
    metrics_dir = (root / e_cfg.get("output_dir", "artifacts/metrics")).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.tune_threshold and val_loader:
        tuned_t, val_acc = tune_threshold_on_val(model, val_loader, device)
        threshold = tuned_t
        logger.info("Tuned threshold on Val set: %.2f (Val Acc: %.4f)", threshold, val_acc)
        with open(metrics_dir / "threshold.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold, "val_accuracy": val_acc}, f, indent=2)

    # Evaluate target split
    all_targets: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []
    post_ids: list[str] = []

    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits = model(b.x, b.edge_index, b.batch)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(b.y.view(-1).cpu().numpy().tolist())
            if hasattr(b, "post_id"):
                post_ids.extend(list(b.post_id))

    metrics = compute_metrics(all_targets, all_preds, all_probs)
    metrics["split"] = args.split
    metrics["threshold"] = threshold
    metrics["checkpoint"] = str(ckpt_path.name)

    # 1. Save metrics JSON
    out_metric_file = metrics_dir / f"{args.split}_metrics.json"
    with open(out_metric_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to: %s", out_metric_file)

    # 2. Plot Confusion Matrix
    figures_dir = (root / e_cfg.get("figures_dir", "artifacts/figures")).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
    logger.info("Confusion matrix figure saved to: %s", figures_dir / "confusion_matrix.png")

    # 3. Plot Reliability Curve
    plot_reliability_curve(all_targets, all_probs, figures_dir / "reliability_curve.png")
    logger.info("Reliability curve figure saved to: %s", figures_dir / "reliability_curve.png")

    # 4. Save Predictions JSONL
    pred_dir = (root / e_cfg.get("predictions_dir", "artifacts/predictions")).resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / f"{args.split}_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for idx in range(len(all_targets)):
            pid = post_ids[idx] if idx < len(post_ids) else f"item_{idx}"
            row = {
                "id": pid,
                "true_label": "FAKE" if all_targets[idx] == 1 else "REAL",
                "predicted_label": "FAKE" if all_preds[idx] == 1 else "REAL",
                "probability_fake": round(float(all_probs[idx]), 4),
                "threshold": threshold,
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Predictions written to: %s", pred_file)

    print("\n" + "=" * 60)
    print(f"CANONICAL EVALUATION RESULTS ({args.split.upper()} SET)")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Decision Threshold: {threshold:.2f}")
    print("=" * 60)
    print(f"Test Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Binary F1-Score:   {metrics['f1']:.4f}")
    print(f"Binary Precision: {metrics['precision']:.4f}")
    print(f"Binary Recall:    {metrics['recall']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC AUC:          {metrics['roc_auc']:.4f}")
    print("=" * 60)
    print(f"Confusion Matrix (Rows=True, Cols=Pred):\n  [REAL, FAKE]")
    print(f"  REAL: {metrics['confusion_matrix'][0]}")
    print(f"  FAKE: {metrics['confusion_matrix'][1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()

