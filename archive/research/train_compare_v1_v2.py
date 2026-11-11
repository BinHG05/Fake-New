"""
Train & Compare v1 (Baseline) vs v2 (CLIP+XLM-R) vs v2b (CLIP-only ablation).
Includes Experiment 4: Expert Fusion (Gated Attention).

Reads features_v2.pt (output of clip_text_extractor_v2.py) and trains
multiple classifiers, printing a final comparison table.

Usage:
    python src/training/train_compare_v1_v2.py
"""

import sys
import time
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import models
from src.models.clip_classifier_v2 import (
    BaselineMLP, 
    CLIPMultimodalMLP, 
    CLIPOnlyMLP, 
    ExpertFusionMLP
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES_PATH = PROJECT_ROOT / "data" / "04_features" / "Fakeddit" / "features_v2.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparison"

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_splits(features: dict):
    """Split data into train/val/test based on the 'splits' field."""
    splits = features["splits"]
    N = len(splits)

    train_idx = [i for i in range(N) if splits[i] == "train"]
    val_idx = [i for i in range(N) if splits[i] == "val"]
    test_idx = [i for i in range(N) if splits[i] == "test"]

    # If no val split exists, carve 15% from train
    if len(val_idx) == 0:
        import random
        random.seed(42)
        random.shuffle(train_idx)
        split_point = int(len(train_idx) * 0.85)
        val_idx = train_idx[split_point:]
        train_idx = train_idx[:split_point]

    print(f"   📊 Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer=None,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    """Train a model and return best validation state."""
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            *inputs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(*inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, targets = [b.to(device) for b in batch]
                logits = model(*inputs)
                val_loss += criterion(logits, targets).item()
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        val_loss /= len(val_loader)
        val_f1 = f1_score(all_targets, all_preds, average="macro")
        val_acc = accuracy_score(all_targets, all_preds)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d} | train_loss={train_loss/len(train_loader):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"      ⏹️  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader, device, name="Model"):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            *inputs, targets = [b.to(device) for b in batch]
            logits = model(*inputs)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    
    report = classification_report(all_targets, all_preds, target_names=["REAL", "FAKE"], digits=4, zero_division=0)
    
    return {
        "name": name,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, 
        "report": report,
        "y_true": all_targets, "y_pred": all_preds, "y_prob": all_probs
    }


def main():
    print("=" * 65)
    print("🏋️  TRAIN & COMPARE: Content Analysis Evolution (v1 -> v2 -> v2 Improved)")
    print("=" * 65)

    if not FEATURES_PATH.exists():
        print(f"\n❌ Features file not found")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")

    features = torch.load(str(FEATURES_PATH), map_location="cpu", weights_only=False)
    train_idx, val_idx, test_idx = prepare_splits(features)

    def make_loaders(tensors, t_i, v_i, ts_i):
        return (
            DataLoader(TensorDataset(*[t[t_i] for t in tensors]), batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(TensorDataset(*[t[v_i] for t in tensors]), batch_size=BATCH_SIZE),
            DataLoader(TensorDataset(*[t[ts_i] for t in tensors]), batch_size=BATCH_SIZE),
        )

    labels = features["labels"]
    
    # --- Experiment 1: v1 Baseline ---
    print("\n🧪 EXPERIMENT 1: v1 Baseline")
    t1 = [features["old_text_feats"], labels]
    l1, v1, ts1 = make_loaders(t1, train_idx, val_idx, test_idx)
    m1 = train_model(BaselineMLP(), l1, v1, device)
    res1 = evaluate_model(m1, ts1, device, "v1 Baseline")

    # --- Experiment 2: v2b CLIP Only ---
    print("\n🧪 EXPERIMENT 2: v2b CLIP-Only Ablation")
    t2b = [features["clip_text_emb"], features["clip_image_emb"], features["clip_score"], labels]
    l2b, v2b, ts2b = make_loaders(t2b, train_idx, val_idx, test_idx)
    m2b = train_model(CLIPOnlyMLP(), l2b, v2b, device)
    res2b = evaluate_model(m2b, ts2b, device, "v2b CLIP-Only")

    # --- Experiment 3: v2 Full (Concatenation) ---
    print("\n🧪 EXPERIMENT 3: v2 Full (CLIP + XLM-R)")
    t2 = [features["clip_text_emb"], features["clip_image_emb"], features["xlmr_emb"], features["clip_score"], labels]
    l2, v2, ts2 = make_loaders(t2, train_idx, val_idx, test_idx)
    m2 = train_model(CLIPMultimodalMLP(), l2, v2, device, lr=2e-4) # Lower LR for stability
    res2 = evaluate_model(m2, ts2, device, "v2 Full CLIP+XLM-R")

    # --- Experiment 4: v2 Improved (Expert Fusion) ---
    print("\n🧪 EXPERIMENT 4: v2 Improved (Gated Expert Fusion)")
    m4 = ExpertFusionMLP().to(device)
    opt4 = torch.optim.AdamW(m4.parameters(), lr=1e-4, weight_decay=2e-3)
    m4 = train_model(m4, l2, v2, device, optimizer=opt4)
    res4 = evaluate_model(m4, ts2, device, "v2 Improved (Expert Fusion)")

    # --- Results Table ---
    all_res = [res1, res2b, res2, res4]
    print("\n\n" + "=" * 70)
    print(f"{'Pipeline':<30} {'Accuracy':>10} {'F1-Score':>10}")
    print("-" * 70)
    for r in all_res:
        print(f"{r['name']:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
    print("=" * 70)

    # Save Best results for visualization
    best_res = max(all_res, key=lambda x: x["accuracy"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "y_true": best_res["y_true"],
        "y_pred": best_res["y_pred"],
        "y_prob": best_res["y_prob"]
    }, RESULTS_DIR / "test_results_v2.pt")
    
    # Save Model
    torch.save(m4.state_dict(), RESULTS_DIR / "best_model_v2.pt")
    
    # Write report file
    with open(RESULTS_DIR / "v1_vs_v2_report.txt", "w", encoding="utf-8") as f:
        f.write("Fake News Detection Report - Content Domain\n\n")
        f.write(f"{'Pipeline':<30} {'Accuracy':>10} {'F1-Score':>10}\n")
        for r in all_res:
            f.write(f"{r['name']:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f}\n")
        for r in all_res:
            f.write(f"\n--- {r['name']} ---\n{r['report']}\n")

    print(f"\n✅ All complete. Visuals ready to generate.")

if __name__ == "__main__":
    main()
