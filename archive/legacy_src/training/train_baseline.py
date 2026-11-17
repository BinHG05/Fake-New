"""
Unified training script for Phase 3 baseline models (Text / Image / Fusion).

Supports:
  --model_type text    → TextOnlyModel
  --model_type image   → ImageOnlyModel
  --model_type fusion  → SimpleFusionModel

Graph baseline uses the existing train_gnn.py instead.

Usage:
  python src/training/train_baseline.py --model_type text  --epochs 20 --lr 2e-5
  python src/training/train_baseline.py --model_type image --epochs 20 --lr 1e-3
  python src/training/train_baseline.py --model_type fusion --epochs 20 --lr 2e-5
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.data.text_image_dataset import (
    create_dataloaders,
    LABEL_MAP_6,
    LABEL_NAMES_6,
    LABEL_NAMES_BINARY,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ==============================================================
# Helpers
# ==============================================================

def build_model(model_type: str, num_classes: int = 2) -> nn.Module:
    """Instantiate the correct model based on --model_type."""
    if model_type == 'text':
        from src.models.baselines.text_only import TextOnlyModel
        return TextOnlyModel(num_classes=num_classes)
    elif model_type == 'image':
        from src.models.baselines.image_only import ImageOnlyModel
        return ImageOnlyModel(num_classes=num_classes)
    elif model_type == 'fusion':
        from src.models.baselines.fusion import SimpleFusionModel
        return SimpleFusionModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'text', 'image', or 'fusion'.")


def get_dataset_mode(model_type: str) -> str:
    """Map model_type → dataset mode ('text', 'image', 'both')."""
    return {'text': 'text', 'image': 'image', 'fusion': 'both'}[model_type]


def forward_batch(model, batch, model_type: str, device: torch.device) -> torch.Tensor:
    """Run a forward pass on one batch, handling different input signatures."""
    if model_type == 'text':
        return model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
        )
    elif model_type == 'image':
        return model(images=batch['image'].to(device))
    elif model_type == 'fusion':
        return model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            images=batch['image'].to(device),
        )


# ==============================================================
# Evaluate
# ==============================================================

@torch.no_grad()
def evaluate(model, loader, model_type, device, split_name='val', label_mode='binary'):
    """Evaluate model on a DataLoader. Returns dict of metrics."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        logits = forward_batch(model, batch, model_type, device)
        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = batch['label'].numpy()
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 6-class metrics
    acc_6 = accuracy_score(all_targets, all_preds)
    f1_macro_6 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # Binary metrics: 0,1,2 → 0 (Real-ish) | 3,4,5 → 1 (Fake-ish)
    preds_bin = (all_preds >= 3).astype(int)
    targets_bin = (all_targets >= 3).astype(int)
    acc_bin = accuracy_score(targets_bin, preds_bin)
    f1_bin = f1_score(targets_bin, preds_bin, average='binary', zero_division=0)

    return {
        f'{split_name}_acc_6': acc_6,
        f'{split_name}_f1_macro_6': f1_macro_6,
        f'{split_name}_acc_bin': acc_bin,
        f'{split_name}_f1_bin': f1_bin,
        '_preds': all_preds,
        '_targets': all_targets,
    }


@torch.no_grad()
def evaluate_binary(model, loader, model_type, device, split_name='val'):
    """Evaluate a binary classifier on a DataLoader."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        logits = forward_batch(model, batch, model_type, device)
        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = batch['label'].numpy()
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc_bin = accuracy_score(all_targets, all_preds)
    f1_bin = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    return {
        f'{split_name}_acc_bin': acc_bin,
        f'{split_name}_f1_bin': f1_bin,
        '_preds': all_preds,
        '_targets': all_targets,
    }


# ==============================================================
# Train
# ==============================================================

def train():
    parser = argparse.ArgumentParser(description='Train Phase-3 Baseline Model')
    parser.add_argument('--model_type', required=True, choices=['text', 'image', 'fusion'],
                        help="Which baseline to train")
    parser.add_argument('--data_path', default=None,
                        help='Path to labeled_master.jsonl (default: auto-detect)')
    parser.add_argument('--label_mode', default='binary', choices=['binary', '6class'],
                        help='Training target mode')
    parser.add_argument('--epochs', type=int, default=20, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--save_dir', default='models/checkpoints', help='Directory to save best model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # ---- Setup ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Label mode: {args.label_mode}")

    # ---- Data ----
    ds_mode = get_dataset_mode(args.model_type)
    logger.info(f"Loading data (mode='{ds_mode}') ...")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        mode=ds_mode,
        label_mode=args.label_mode,
        batch_size=args.batch_size,
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # ---- Class weights ----
    train_labels = []
    for batch in train_loader:
        train_labels.append(batch['label'].numpy())
    train_labels = np.concatenate(train_labels)

    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    num_classes = 2 if args.label_mode == 'binary' else 6
    full_weights = torch.ones(num_classes, device=device)
    for i, c in enumerate(classes):
        full_weights[int(c)] = weights[i]
    logger.info(f"Class weights: {full_weights.tolist()}")

    # ---- Model ----
    logger.info("Building model ...")
    model = build_model(args.model_type, num_classes=num_classes).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ---- Optimizer & Loss ----
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=full_weights)

    # ---- Training loop ----
    best_val_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(args.save_dir, f'best_{args.model_type}_baseline_{args.label_mode}.pt')

    logger.info("=" * 50)
    logger.info("Starting Training")
    logger.info("=" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            labels = batch['label'].to(device)
            logits = forward_batch(model, batch, args.model_type, device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Evaluate on val
        if args.label_mode == 'binary':
            val_metrics = evaluate_binary(model, val_loader, args.model_type, device, 'val')
            current_f1 = val_metrics['val_f1_bin']
            logger.info(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val F1(bin): {current_f1:.4f} | "
                f"Val Acc(bin): {val_metrics['val_acc_bin']:.4f} | "
                f"{elapsed:.1f}s"
            )
        else:
            val_metrics = evaluate(model, val_loader, args.model_type, device, 'val', args.label_mode)
            current_f1 = val_metrics['val_f1_macro_6']
            logger.info(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Val F1(macro): {current_f1:.4f} | "
                f"Val Acc(bin): {val_metrics['val_acc_bin']:.4f} | "
                f"{elapsed:.1f}s"
            )

        # Early stopping
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✅ Saved best model (F1={current_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ---- Final Test ----
    logger.info("=" * 50)
    logger.info("Loading best model for final test ...")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    if args.label_mode == 'binary':
        test_metrics = evaluate_binary(model, test_loader, args.model_type, device, 'test')
    else:
        test_metrics = evaluate(model, test_loader, args.model_type, device, 'test', args.label_mode)

    print("\n" + "=" * 40)
    print(f"FINAL TEST RESULTS — {args.model_type.upper()} BASELINE")
    print("=" * 40)
    if args.label_mode == 'binary':
        print(f"Binary Accuracy:   {test_metrics['test_acc_bin']:.4f}")
        print(f"Binary F1:         {test_metrics['test_f1_bin']:.4f}")
    else:
        print(f"6-Class Accuracy:  {test_metrics['test_acc_6']:.4f}")
        print(f"6-Class Macro-F1:  {test_metrics['test_f1_macro_6']:.4f}")
        print(f"Binary Accuracy:   {test_metrics['test_acc_bin']:.4f}")
        print(f"Binary F1:         {test_metrics['test_f1_bin']:.4f}")
    print("=" * 40)

    # Classification report
    print(f"\nClassification Report ({'Binary' if args.label_mode == 'binary' else '6-Class'}):")
    present_classes = sorted(set(test_metrics['_targets']) | set(test_metrics['_preds']))
    if args.label_mode == 'binary':
        present_names = [LABEL_NAMES_BINARY[i] for i in present_classes]
    else:
        present_names = [LABEL_NAMES_6[i] for i in present_classes]
    print(classification_report(
        test_metrics['_targets'],
        test_metrics['_preds'],
        target_names=present_names,
        labels=present_classes,
        zero_division=0,
    ))


if __name__ == '__main__':
    train()
