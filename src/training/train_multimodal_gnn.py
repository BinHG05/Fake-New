"""
Training script for Phase 4: Multimodal GNN on Cascade Graphs.

Loads individual cascade graphs from data/processed_graphs_multimodal/
and applies CrossAttention fusion + GNN for graph-level classification.

Usage:
  # Full model (cross-attention + GNN)
  python src/training/train_multimodal_gnn.py --epochs 50 --lr 1e-3

  # Gated fusion variant
  python src/training/train_multimodal_gnn.py --fusion_type gated --epochs 50

  # Ablation: text only (zero image features)
  python src/training/train_multimodal_gnn.py --ablation text_only

  # Ablation: no graph (fusion only, skip GNN)
  python src/training/train_multimodal_gnn.py --ablation no_graph
"""

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from src.data.label_utils import BINARY_LABEL_MAP, BINARY_LABEL_NAMES, derive_binary_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_NAMES = BINARY_LABEL_NAMES


# =============================================================================
# Fusion Modules (inline to keep self-contained)
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """Text queries Image via multi-head attention with gated residual."""

    def __init__(self, text_dim, image_dim, fusion_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        assert fusion_dim % num_heads == 0
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate_linear = nn.Linear(fusion_dim * 2, 1)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb)
        i = self.image_proj(image_emb)
        attended, _ = self.attn(t.unsqueeze(1), i.unsqueeze(1), i.unsqueeze(1))
        attended = attended.squeeze(1)
        gate = torch.sigmoid(self.gate_linear(torch.cat([t, attended], dim=-1)))
        return self.norm(gate * attended + (1 - gate) * t)


class GatedFusion(nn.Module):
    """Simple gated fusion between two modalities."""

    def __init__(self, text_dim, image_dim, fusion_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.gate_linear = nn.Linear(fusion_dim * 2, 1)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_emb, image_emb):
        t = F.relu(self.text_proj(text_emb))
        i = F.relu(self.image_proj(image_emb))
        gate = torch.sigmoid(self.gate_linear(torch.cat([t, i], dim=-1)))
        return self.norm(gate * t + (1 - gate) * i)


# =============================================================================
# Multimodal Cascade GNN
# =============================================================================

class MultimodalCascadeGNN(nn.Module):
    """
    Multimodal GNN for graph-level classification on cascade graphs.

    Architecture:
      node features (768d text) → split into text/image OR use as-is
      → Fusion module (if image features available)
      → GNN message passing
      → Global mean pool
      → MLP classifier → [B, 6]
    """

    def __init__(
        self,
        input_dim: int = 768,
        text_dim: int = 768,
        image_dim: int = 0,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        gnn_type: str = 'sage',
        fusion_type: str = 'cross_attention',
    ):
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.has_image = image_dim > 0
        self.dropout_rate = dropout

        # Fusion or projection
        if self.has_image:
            if fusion_type == 'cross_attention':
                self.fusion = CrossAttentionFusion(text_dim, image_dim, fusion_dim)
            else:
                self.fusion = GatedFusion(text_dim, image_dim, fusion_dim)
            gnn_input_dim = fusion_dim
        else:
            # Text-only: just project down
            self.input_proj = nn.Linear(input_dim, fusion_dim)
            gnn_input_dim = fusion_dim

        # GNN
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = gnn_input_dim if i == 0 else hidden_dim
            if gnn_type == 'sage':
                self.convs.append(SAGEConv(in_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(in_dim, hidden_dim, heads=1))
            else:
                self.convs.append(GCNConv(in_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Residual proj for first layer
        self.residual_proj = nn.Linear(gnn_input_dim, hidden_dim) if gnn_input_dim != hidden_dim else nn.Identity()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, ablation=None):
        """
        Args:
            x: [total_nodes, input_dim] node features
            edge_index: [2, E]
            batch: [total_nodes] graph assignment
            ablation: 'text_only', 'no_graph', or None
        """
        if self.has_image:
            text_emb = x[:, :self.text_dim]
            image_emb = x[:, self.text_dim:]

            if ablation == 'text_only':
                image_emb = torch.zeros_like(image_emb)
            elif ablation == 'image_only':
                text_emb = torch.zeros_like(text_emb)

            h = self.fusion(text_emb, image_emb)
        else:
            h = F.relu(self.input_proj(x))

        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GNN (skip if ablation='no_graph')
        if ablation != 'no_graph' and edge_index.numel() > 0:
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)

                if i == 0:
                    h = h_new + self.residual_proj(h)
                else:
                    h = h_new + h

        # Pool → graph-level
        graph_emb = global_mean_pool(h, batch)
        return self.classifier(graph_emb)


# =============================================================================
# Data Loading (reuse from train_gnn.py pattern)
# =============================================================================

def load_cascade_graphs(graph_dir, metadata_path):
    logger.info(f"📄 Loading metadata from: {metadata_path}")
    id_to_meta = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            try:
                binary_label, _ = derive_binary_label(item)
            except ValueError:
                continue

            id_to_meta[item['id']] = {
                'label': BINARY_LABEL_MAP[binary_label],
                'split': item['split']
            }
    logger.info(f"  Found {len(id_to_meta)} labeled entries")

    torch.serialization.add_safe_globals([Data])
    graph_dir = Path(graph_dir)
    train_graphs, val_graphs, test_graphs = [], [], []
    loaded, skipped = 0, 0

    for pt_file in sorted(graph_dir.glob('*.pt')):
        post_id = pt_file.stem
        if post_id not in id_to_meta:
            skipped += 1
            continue
        try:
            data = torch.load(pt_file, weights_only=False)
            meta = id_to_meta[post_id]
            data.y = torch.tensor([meta['label']], dtype=torch.long)

            if data.edge_index.numel() == 0:
                data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)

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

    logger.info(f"✅ Loaded {loaded} graphs (skipped {skipped})")
    logger.info(f"  Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
    return train_graphs, val_graphs, test_graphs


# =============================================================================
# Evaluate
# =============================================================================

def evaluate(model, loader, device, ablation=None, split='val'):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch, ablation=ablation)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    if targets.size == 0:
        return {
            f'{split}_acc_bin': 0.0,
            f'{split}_f1_bin': 0.0,
            '_preds': preds, '_targets': targets,
        }

    acc_bin = accuracy_score(targets, preds)
    f1_bin = f1_score(targets, preds, average='binary', zero_division=0)

    return {
        f'{split}_acc_bin': acc_bin,
        f'{split}_f1_bin': f1_bin,
        '_preds': preds, '_targets': targets,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4: Multimodal GNN on Cascade Graphs')

    # Data
    parser.add_argument('--graph_dir', default='data/processed_graphs_multimodal')
    parser.add_argument('--metadata', default='data/reddit_enriched_binary.jsonl')

    # Model
    parser.add_argument('--fusion_type', default='cross_attention', choices=['cross_attention', 'gated'])
    parser.add_argument('--fusion_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--gnn_type', default='sage', choices=['sage', 'gcn', 'gat'])
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_dir', default='models/checkpoints')
    parser.add_argument('--seed', type=int, default=42)

    # Ablation
    parser.add_argument('--ablation', default=None, choices=[None, 'text_only', 'image_only', 'no_graph'])

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Fusion: {args.fusion_type} | GNN: {args.gnn_type} | Ablation: {args.ablation or 'FULL'}")

    # Load data
    train_graphs, val_graphs, test_graphs = load_cascade_graphs(args.graph_dir, args.metadata)
    if not train_graphs:
        logger.error("No training graphs found!")
        return

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    # Determine feature dims
    input_dim = train_graphs[0].x.size(1)
    # Detect if features are text-only (768) or text+image concat
    if input_dim > 768:
        text_dim = 768
        image_dim = input_dim - 768
    else:
        text_dim = input_dim
        image_dim = 0

    logger.info(f"Input dim: {input_dim} (text={text_dim}, image={image_dim})")

    # Class weights
    y_train = np.array([g.y.item() for g in train_graphs])
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    full_weights = torch.ones(args.num_classes, device=device)
    for i, c in enumerate(classes):
        full_weights[int(c)] = weights[i]
    logger.info(f"Class weights: {full_weights.tolist()}")

    # Model
    model = MultimodalCascadeGNN(
        input_dim=input_dim,
        text_dim=text_dim,
        image_dim=image_dim,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.fusion_dim, # Changed to fusion_dim for consistency with GNN input
        num_classes=args.num_classes,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        fusion_type=args.fusion_type,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable:,} trainable")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=full_weights)

    # Train
    best_val_f1 = 0.0
    patience_counter = 0
    save_name = f'best_multimodal_cascade_binary_{args.fusion_type}'
    if args.ablation:
        save_name += f'_{args.ablation}'
    save_path = os.path.join(args.save_dir, f'{save_name}.pt')

    logger.info("=" * 50)
    logger.info("Starting Training")
    logger.info("=" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, num_batches = 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, ablation=args.ablation)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        val_m = evaluate(model, val_loader, device, args.ablation, 'val')

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                f"Val F1(bin): {val_m['val_f1_bin']:.4f} | "
                f"Val Acc(bin): {val_m['val_acc_bin']:.4f}"
            )

        if val_m['val_f1_bin'] > best_val_f1:
            best_val_f1 = val_m['val_f1_bin']
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test
    logger.info("Loading best model for final test ...")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_m = evaluate(model, test_loader, device, args.ablation, 'test')

    ablation_label = (args.ablation or 'FULL').upper()
    print(f"\n{'=' * 50}")
    print(f"FINAL TEST — MULTIMODAL CASCADE GNN ({ablation_label})")
    print(f"Fusion: {args.fusion_type} | GNN: {args.gnn_type} × {args.num_gnn_layers}")
    print(f"{'=' * 50}")
    print(f"Binary Accuracy:   {test_m['test_acc_bin']:.4f}")
    print(f"Binary F1:         {test_m['test_f1_bin']:.4f}")
    print(f"{'=' * 50}")

    present = sorted(set(test_m['_targets']) | set(test_m['_preds']))
    print("\nClassification Report:")
    print(classification_report(
        test_m['_targets'], test_m['_preds'],
        target_names=[LABEL_NAMES[i] for i in present],
        labels=present, zero_division=0,
    ))

    # Comparison
    print(f"\n{'-' * 50}")
    print("VS PHASE 3 BASELINES")
    print(f"{'-' * 50}")
    print(f"{'Model':<30} {'Binary F1':>10} {'Binary Acc':>10}")
    print(f"{'-' * 50}")
    print(f"{'Text-Only (Phase 3)':30} {'0.5250':>10} {'0.5250':>10}")
    print(f"{'Graph-Only GCN (Phase 3)':30} {'0.6320':>10} {'0.6320':>10}")
    print(f"{'Multimodal GNN (' + ablation_label + ')':30} {test_m['test_f1_bin']:>10.4f} {test_m['test_acc_bin']:>10.4f}")
    delta_f1 = test_m['test_f1_bin'] - 0.632
    delta_acc = test_m['test_acc_bin'] - 0.632
    print(f"{'Delta vs best baseline':30} {delta_f1:>+10.4f} {delta_acc:>+10.4f}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
