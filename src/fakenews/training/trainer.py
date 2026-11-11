"""
Modular Trainer for Multimodal Cascade GNN.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from src.fakenews.config.settings import get_project_root
from src.fakenews.evaluation.metrics import compute_metrics
from src.fakenews.models.multimodal_gnn import FocalLoss, MultimodalCascadeGNN
from src.fakenews.training.dataset import AugmentedGraphDataset, load_cascade_graphs

logger = logging.getLogger(__name__)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Runs inference across a DataLoader and returns evaluation metrics."""
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            targets = batch.y.view(-1).cpu().numpy()

            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    return compute_metrics(all_targets, all_preds, all_probs)


def train_pipeline(
    config: dict[str, Any],
    epochs_override: int | None = None,
    limit_override: int | None = None,
) -> dict[str, Any]:
    """
    Executes the training pipeline according to the provided config dictionary.
    Returns dictionary with final metrics and checkpoint paths.
    """
    root = get_project_root()
    ds_cfg = config.get("dataset", {})
    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})
    e_cfg = config.get("evaluation", {})

    seed = int(t_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s (Seed: %d)", device, seed)

    # Resolve paths
    meta_path = root / ds_cfg.get("metadata", "data/all_merged_binary.jsonl")
    graph_dir = root / ds_cfg.get("graph_dir", "data/processed_graphs_v3")

    limit = limit_override if limit_override is not None else t_cfg.get("limit")
    train_graphs, val_graphs, test_graphs = load_cascade_graphs(
        graph_dir=graph_dir,
        metadata_path=meta_path,
        limit=limit,
    )

    if not train_graphs:
        raise RuntimeError(f"No training graphs found in {graph_dir}")

    logger.info(
        "Dataset loaded: Train=%d | Val=%d | Test=%d",
        len(train_graphs),
        len(val_graphs),
        len(test_graphs),
    )

    batch_size = int(t_cfg.get("batch_size", 32))
    train_loader = DataLoader(
        AugmentedGraphDataset(train_graphs, base_seed=seed),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False) if val_graphs else None
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False) if test_graphs else None

    # Model dimensions
    text_dim = int(m_cfg.get("text_dim", 768))
    image_dim = int(m_cfg.get("image_dim", 1025))
    input_dim = text_dim + image_dim
    fusion_dim = int(m_cfg.get("fusion_dim", 512))
    hidden_dim = int(m_cfg.get("hidden_dim", 512))
    num_classes = int(m_cfg.get("num_classes", 2))
    num_gnn_layers = int(m_cfg.get("num_gnn_layers", 3))
    dropout = float(m_cfg.get("dropout", 0.3))
    fusion = str(m_cfg.get("fusion", "cross_attention"))
    gnn_type = str(m_cfg.get("gnn", "gat"))
    learn_missing = bool(m_cfg.get("learn_missing_image_emb", False))

    model = MultimodalCascadeGNN(
        input_dim=input_dim,
        text_dim=text_dim,
        image_dim=image_dim,
        fusion_dim=fusion_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
        gnn_type=gnn_type,
        fusion_type=fusion,
        learn_missing_image_emb=learn_missing,
    ).to(device)

    lr = float(t_cfg.get("learning_rate", 1e-4))
    wd = float(t_cfg.get("weight_decay", 1e-3))
    fake_weight = float(t_cfg.get("fake_class_weight", 2.4))
    smoothing = float(t_cfg.get("label_smoothing", 0.05))
    gamma = float(t_cfg.get("focal_gamma", 2.0))

    class_weights = torch.tensor([1.0, fake_weight], device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=gamma, smoothing=smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = epochs_override if epochs_override is not None else int(t_cfg.get("epochs", 50))
    patience = int(t_cfg.get("patience", 15))
    selection_metric = str(t_cfg.get("selection_metric", "acc"))

    checkpoint_dir = root / t_cfg.get("checkpoint_dir", "artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(t_cfg.get("checkpoint_name", "multimodal_cascade_gnn_binary.pt"))
    checkpoint_path = checkpoint_dir / checkpoint_name

    metrics_dir = root / e_cfg.get("output_dir", "artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("-inf")
    patience_counter = 0
    train_history = []

    logger.info("Starting training loop for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        val_metrics = {}
        if val_loader:
            val_metrics = evaluate_loader(model, val_loader, device)
            score = val_metrics["accuracy"] if selection_metric == "acc" else val_metrics["f1"]
            if score >= best_score:
                best_score = score
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(
                    "Epoch %02d | Loss: %.4f | Val Acc: %.4f | Val F1: %.4f -> (New Best, Saved Checkpoint)",
                    epoch,
                    avg_loss,
                    val_metrics["accuracy"],
                    val_metrics["f1"],
                )
            else:
                patience_counter += 1
                logger.info(
                    "Epoch %02d | Loss: %.4f | Val Acc: %.4f | Val F1: %.4f (Patience %d/%d)",
                    epoch,
                    avg_loss,
                    val_metrics["accuracy"],
                    val_metrics["f1"],
                    patience_counter,
                    patience,
                )
        else:
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Epoch %02d | Loss: %.4f", epoch, avg_loss)

        train_history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_metrics": val_metrics,
        })

        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # Save training metrics history
    with open(metrics_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"epochs": epochs, "best_val_score": best_score, "history": train_history},
            f,
            indent=2,
        )

    # Evaluate best checkpoint on test set
    if test_loader and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        test_metrics = evaluate_loader(model, test_loader, device)
        with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(
            "Final Test Evaluation: Accuracy=%.4f, F1=%.4f",
            test_metrics["accuracy"],
            test_metrics["f1"],
        )
    else:
        test_metrics = {}

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_score": best_score,
        "test_metrics": test_metrics,
    }

