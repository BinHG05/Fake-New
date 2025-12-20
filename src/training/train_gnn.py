"""
Training script for MultiModal Fake News GNN on Interaction Graph.

Features:
- 6-class classification.
- Binary evaluation monitoring.
- Early Stopping based on Validation Macro-F1.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report
import argparse
from pathlib import Path
import logging
from typing import Dict

# Internal imports
from src.data.dataloader import FakeNewsGraphDataset
from src.models.cascade_gnn import MultiModalFakeNewsGNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(model, data, mask, split_name="val") -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        
        # 6-class metrics
        preds_6 = logits[mask].argmax(dim=-1).cpu().numpy()
        targets_6 = data.y[mask].cpu().numpy()
        
        acc_6 = accuracy_score(targets_6, preds_6)
        f1_macro_6 = f1_score(targets_6, preds_6, average='macro')
        
        # Binary metrics
        # Map 6-class to binary: 0,1,2 -> 0 (True), 3,4,5 -> 1 (Fake)
        preds_bin = (preds_6 >= 3).astype(int)
        targets_bin = (targets_6 >= 3).astype(int)
        
        acc_bin = accuracy_score(targets_bin, preds_bin)
        f1_bin = f1_score(targets_bin, preds_bin, average='binary')
        
    return {
        f'{split_name}_acc_6': acc_6,
        f'{split_name}_f1_macro_6': f1_macro_6,
        f'{split_name}_acc_bin': acc_bin,
        f'{split_name}_f1_bin': f1_bin
    }

def train():
    parser = argparse.ArgumentParser(description='Train GNN on Interaction Graph')
    parser.add_argument('--graph', default='data/04_graph/fakeddit_graph.pt', help='Path to graph .pt file')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--gnn_type', choices=['gat', 'sage', 'gcn'], default='gat', help='GNN layer type')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--save_dir', default='models/checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset (Note: weights_only=False is handled inside dataloader or here)
    # Since PyTorch 2.6 requires add_safe_globals for Data, we do it here
    from torch_geometric.data import Data
    torch.serialization.add_safe_globals([Data])
    
    dataset = FakeNewsGraphDataset(graph_path=args.graph)
    data = dataset.graph.to(device)
    
    # Initialize model
    model = MultiModalFakeNewsGNN(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=6,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    ).to(device)
    
    # Calculate class weights for imbalance
    y_train = data.y[data.train_mask].cpu().numpy()
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # Ensure weight vector covers all 6 classes even if some are missing in train_mask
    full_weights = torch.ones(6).to(device)
    for i, c in enumerate(classes):
        full_weights[c] = weights[i]
    
    logger.info(f"Class weights: {full_weights.tolist()}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=full_weights)
    
    best_val_f1 = 0
    patience_counter = 0
    
    logger.info("Starting Training...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluate
        val_metrics = evaluate(model, data, data.val_mask, "val")
        current_val_f1 = val_metrics['val_f1_macro_6']
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1 (Macro): {current_val_f1:.4f} | Val Acc (Bin): {val_metrics['val_acc_bin']:.4f}")
            
        # Early Stopping based on Val Macro-F1
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            patience_counter = 0
            # Save best model
            save_path = os.path.join(args.save_dir, 'best_gnn_model.pt')
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
    # Final Test Evaluation
    logger.info("Training complete. Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_gnn_model.pt'), weights_only=False))
    
    test_metrics = evaluate(model, data, data.test_mask, "test")
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"6-Class Accuracy:  {test_metrics['test_acc_6']:.4f}")
    print(f"6-Class Macro-F1:  {test_metrics['test_f1_macro_6']:.4f}")
    print(f"Binary Accuracy:   {test_metrics['test_acc_bin']:.4f}")
    print(f"Binary F1:         {test_metrics['test_f1_bin']:.4f}")
    print("="*30)
    
    # Detailed report
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=-1).cpu().numpy()
        targets = data.y[data.test_mask].cpu().numpy()
        
        print("\nClassification Report (6-Class):")
        target_names = ['TRUE', 'MOSTLY_TRUE', 'HALF_TRUE', 'BARELY_TRUE', 'FALSE', 'PANTS_ON_FIRE']
        # Map target names present in this test split
        present_classes = sorted(list(set(targets) | set(preds)))
        present_names = [target_names[i] for i in present_classes]
        print(classification_report(targets, preds, target_names=present_names, labels=present_classes))

if __name__ == "__main__":
    train()