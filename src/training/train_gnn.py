"""
Training script for MultiModal Fake News GNN on Cascade Graphs.

Features:
- Loads individual cascade graphs from data/processed_graphs/
- Maps labels from data/04_graph/merged_data.jsonl
- Graph-level classification (each cascade = 1 sample)
- 6-class classification with binary evaluation monitoring
- Early Stopping based on Validation Macro-F1
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv, SAGEConv, GATConv
from sklearn.metrics import f1_score, accuracy_score, classification_report
import argparse
import logging
import numpy as np
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Label Mapping ───
LABEL_MAP = {
    'TRUE': 0,
    'MOSTLY_TRUE': 1,
    'HALF_TRUE': 2,
    'BARELY_TRUE': 3,
    'FALSE': 4,
    'PANTS_ON_FIRE': 5
}
LABEL_NAMES = ['TRUE', 'MOSTLY_TRUE', 'HALF_TRUE', 'BARELY_TRUE', 'FALSE', 'PANTS_ON_FIRE']


# ─── GNN Model for Graph-Level Classification ───
class CascadeGNN(nn.Module):
    """
    GNN for graph-level classification on cascade graphs.
    Uses message passing layers + global pooling → MLP classifier.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 6,
                 dropout: float = 0.3, gnn_type: str = 'gcn', num_layers: int = 2):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: input_dim → hidden_dim
        self.convs.append(self._make_conv(input_dim, hidden_dim, gnn_type))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim, gnn_type))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Input projection for residual (only if dims differ)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Classifier (after global pooling)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def _make_conv(self, in_dim, out_dim, gnn_type):
        if gnn_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif gnn_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        elif gnn_type == 'gat':
            return GATConv(in_dim, out_dim, heads=1)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    def forward(self, x, edge_index, batch):
        # Message passing
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            if i == 0:
                h = h_new + self.input_proj(h)
            else:
                h = h_new + h
            
        # Global pooling: aggregate node features → graph-level
        graph_emb = global_mean_pool(h, batch)  # [num_graphs, hidden_dim]
        
        # Classify
        logits = self.classifier(graph_emb)
        return logits


# ─── Data Loading ───
def load_cascade_graphs(graph_dir: str, metadata_path: str) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Load individual cascade graphs and split into train/val/test.
    
    Args:
        graph_dir: Path to directory with individual .pt graph files
        metadata_path: Path to merged_data.jsonl with labels and splits
        
    Returns:
        train_graphs, val_graphs, test_graphs
    """
    # Load metadata
    logger.info(f"📄 Loading metadata from: {metadata_path}")
    id_to_meta = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            id_to_meta[item['id']] = {
                'label': LABEL_MAP[item['label']],
                'split': item['split']
            }
    logger.info(f"  Found {len(id_to_meta)} labeled entries")
    
    # Allow safe deserialization of Data objects
    torch.serialization.add_safe_globals([Data])
    
    # Load graphs
    graph_dir = Path(graph_dir)
    train_graphs, val_graphs, test_graphs = [], [], []
    loaded, skipped = 0, 0
    
    for pt_file in sorted(graph_dir.glob('*.pt')):
        post_id = pt_file.stem  # filename without extension
        
        if post_id not in id_to_meta:
            skipped += 1
            continue
        
        try:
            data = torch.load(pt_file, weights_only=False)
            
            # Assign label
            meta = id_to_meta[post_id]
            data.y = torch.tensor([meta['label']], dtype=torch.long)
            
            # Handle edge_index for isolated nodes (no edges)
            if data.edge_index.numel() == 0:
                # Create self-loop for root node so GNN can still process
                data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            # Route to correct split
            split = meta['split']
            if split == 'train':
                train_graphs.append(data)
            elif split == 'val':
                val_graphs.append(data)
            elif split == 'test':
                test_graphs.append(data)
            
            loaded += 1
        except Exception as e:
            logger.warning(f"Failed to load {pt_file.name}: {e}")
            skipped += 1
    
    logger.info(f"✅ Loaded {loaded} graphs (skipped {skipped} without labels)")
    logger.info(f"  Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
    
    return train_graphs, val_graphs, test_graphs


def evaluate(model, loader, device, split_name="val") -> Dict[str, float]:
    """Evaluate model on a DataLoader."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            targets = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 6-class metrics
    acc_6 = accuracy_score(all_targets, all_preds)
    f1_macro_6 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Binary metrics: 0,1,2 → 0 (True-ish), 3,4,5 → 1 (Fake-ish)
    preds_bin = (all_preds >= 3).astype(int)
    targets_bin = (all_targets >= 3).astype(int)
    
    acc_bin = accuracy_score(targets_bin, preds_bin)
    f1_bin = f1_score(targets_bin, preds_bin, average='binary', zero_division=0)
    
    return {
        f'{split_name}_acc_6': acc_6,
        f'{split_name}_f1_macro_6': f1_macro_6,
        f'{split_name}_acc_bin': acc_bin,
        f'{split_name}_f1_bin': f1_bin
    }


def train():
    parser = argparse.ArgumentParser(description='Train GNN on Cascade Graphs')
    parser.add_argument('--graph_dir', default='data/processed_graphs', help='Directory with individual .pt graph files')
    parser.add_argument('--metadata', default='data/reddit_enriched_data.jsonl', help='Path to reddit_enriched_data.jsonl')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--gnn_type', choices=['gat', 'sage', 'gcn'], default='gat', help='GNN layer type')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--save_dir', default='models/checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ─── Load Data ───
    train_graphs, val_graphs, test_graphs = load_cascade_graphs(args.graph_dir, args.metadata)
    
    if len(train_graphs) == 0:
        logger.error("No training graphs found! Check graph_dir and metadata paths.")
        return
    
    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Determine input dimension from first graph
    input_dim = train_graphs[0].x.size(1)
    logger.info(f"Input feature dimension: {input_dim}")
    
    # ─── Model ───
    model = CascadeGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=6,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        num_layers=args.num_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: CascadeGNN ({args.gnn_type.upper()}) | Params: {total_params:,}")
    
    # ─── Class Weights ───
    y_train = np.array([g.y.item() for g in train_graphs])
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
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
        total_loss = 0
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
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, "val")
        current_val_f1 = val_metrics['val_f1_macro_6']
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                f"Val F1 (Macro): {current_val_f1:.4f} | "
                f"Val Acc (Bin): {val_metrics['val_acc_bin']:.4f}"
            )
            
        # Early Stopping based on Val Macro-F1
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            patience_counter = 0
            save_path = os.path.join(args.save_dir, f'best_cascade_gnn_{args.gnn_type}.pt')
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
    # ─── Final Test ───
    logger.info("Training complete. Loading best model for testing...")
    best_path = os.path.join(args.save_dir, f'best_cascade_gnn_{args.gnn_type}.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
    
    test_metrics = evaluate(model, test_loader, device, "test")
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"GNN Type:          {args.gnn_type.upper()}")
    print(f"6-Class Accuracy:  {test_metrics['test_acc_6']:.4f}")
    print(f"6-Class Macro-F1:  {test_metrics['test_f1_macro_6']:.4f}")
    print(f"Binary Accuracy:   {test_metrics['test_acc_bin']:.4f}")
    print(f"Binary F1:         {test_metrics['test_f1_bin']:.4f}")
    print("="*30)
    
    # Detailed report
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    print("\nClassification Report (6-Class):")
    present_classes = sorted(list(set(all_targets) | set(all_preds)))
    present_names = [LABEL_NAMES[i] for i in present_classes]
    print(classification_report(all_targets, all_preds, target_names=present_names, labels=present_classes))


if __name__ == "__main__":
    train()