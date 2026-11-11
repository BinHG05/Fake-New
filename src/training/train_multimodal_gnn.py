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


class FocalLoss(nn.Module):
    """Focal Loss with Label Smoothing."""
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)

        log_pt = F.log_softmax(inputs, dim=-1)
        ce_loss = torch.sum(-true_dist * log_pt, dim=-1)
        
        # Original PT for focal scaling
        pt = torch.exp(F.log_softmax(inputs, dim=-1).gather(1, targets.unsqueeze(1)).squeeze())
        
        # If alpha is provided, apply it based on targets
        if self.alpha is not None:
            at = self.alpha.gather(0, targets.data)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class MultimodalCascadeGNN(nn.Module):
    """
    Enhanced Multimodal GNN V4.
    Architecture:
      Fusion (CrossAttention/Gated) -> GATv2 layers -> Hybrid Global Pooling (Mean+Max) -> MLP
    """

    def __init__(
        self,
        input_dim: int = 768,
        text_dim: int = 768,
        image_dim: int = 0,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_gnn_layers: int = 3,
        dropout: float = 0.3,
        gnn_type: str = 'sage',
        fusion_type: str = 'cross_attention',
        learn_missing_image_emb: bool = False,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.has_image = image_dim > 0
        self.dropout_rate = dropout
        self.gnn_type = gnn_type
        self.learn_missing_image_emb = bool(learn_missing_image_emb)

        # Fusion or projection
        if self.has_image:
            if self.learn_missing_image_emb:
                self.missing_image_emb = nn.Parameter(torch.randn(image_dim) * 0.02)
            if fusion_type == 'cross_attention':
                self.fusion = CrossAttentionFusion(text_dim, image_dim, fusion_dim)
            else:
                self.fusion = GatedFusion(text_dim, image_dim, fusion_dim)
            self.fusion_type = fusion_type
            gnn_input_dim = fusion_dim
        else:
            # Text-only: just project down
            self.input_proj = nn.Linear(input_dim, fusion_dim)
            gnn_input_dim = fusion_dim

        # Default to the proven 3-layer setup, but still honor explicit overrides.
        from torch_geometric.nn import GATv2Conv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        effective_layers = max(1, int(num_gnn_layers))
        for i in range(effective_layers):
            in_dim = gnn_input_dim if i == 0 else hidden_dim
            self.convs.append(GATv2Conv(in_dim, hidden_dim // 8, heads=8, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Residual proj for first layer
        self.residual_proj = nn.Linear(gnn_input_dim, hidden_dim) if gnn_input_dim != hidden_dim else nn.Identity()

        # Classifier — input: (hidden_dim*2 from Mean+Max) + (gnn_input_dim from Skip)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + gnn_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, ablation=None, return_attention=False):
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
            elif ablation == 'missing_image':
                image_emb = torch.zeros_like(image_emb)

            if self.learn_missing_image_emb:
                missing_mask = image_emb.abs().sum(dim=-1, keepdim=True) == 0
                if missing_mask.any():
                    image_emb = torch.where(
                        missing_mask,
                        self.missing_image_emb.unsqueeze(0).expand_as(image_emb),
                        image_emb,
                    )
            h = self.fusion(text_emb, image_emb)
        else:
            h = F.relu(self.input_proj(x))

        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        # GNN Forward with Residual Connections
        curr_h = h
        layer_attentions = []
        if ablation != 'no_graph':
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                if return_attention:
                    h_new, (ei, alpha) = conv(curr_h, edge_index, return_attention_weights=True)
                    layer_attentions.append({
                        'edge_index': ei,
                        'alpha': alpha,
                    })
                else:
                    h_new = conv(curr_h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)

                if i == 0:
                    curr_h = h_new + self.residual_proj(curr_h)
                else:
                    curr_h = h_new + curr_h

        # Hybrid Pooling (Mean + Max)
        from torch_geometric.nn import global_max_pool, global_mean_pool
        mean_p = global_mean_pool(curr_h, batch)
        max_p = global_max_pool(curr_h, batch)
        graph_emb = torch.cat([mean_p, max_p], dim=-1)  # [B, hidden_dim * 2]
        
        # GLOBAL SKIP CONNECTION
        skip_emb = global_mean_pool(h, batch)  # [B, gnn_input_dim]
        
        final_emb = torch.cat([graph_emb, skip_emb], dim=-1)
        logits = self.classifier(final_emb)
        if return_attention:
            return logits, {
                'layer_attentions': layer_attentions,
                'node_embeddings': curr_h.detach(),
            }
        return logits


def augment_graph(graph, node_drop_prob=0.0, edge_drop_prob=0.0, feature_mask_prob=0.0, seed=None):
    num_nodes = int(graph.num_nodes)
    if num_nodes <= 1:
        return graph

    rng = np.random.default_rng(seed)
    keep_nodes = np.arange(num_nodes)

    if node_drop_prob > 0:
        # Always keep the root node (index 0)
        comment_nodes = np.arange(1, num_nodes)
        keep_comment_mask = rng.random(comment_nodes.shape[0]) >= node_drop_prob
        keep_nodes = np.concatenate(([0], comment_nodes[keep_comment_mask]))
        keep_nodes = np.unique(keep_nodes)

    node_map = {int(old): new for new, old in enumerate(keep_nodes.tolist())}
    new_x = graph.x[keep_nodes].clone()

    new_edge_index = graph.edge_index
    if edge_drop_prob > 0 or len(keep_nodes) != num_nodes:
        edges = graph.edge_index.detach().cpu().numpy().T
        filtered_edges = []
        for src, dst in edges:
            src_i, dst_i = int(src), int(dst)
            if src_i in node_map and dst_i in node_map:
                if edge_drop_prob <= 0 or rng.random() >= edge_drop_prob:
                    filtered_edges.append([node_map[src_i], node_map[dst_i]])
        
        if filtered_edges:
            new_edge_index = torch.tensor(filtered_edges, dtype=torch.long).T.contiguous()
        else:
            new_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    if feature_mask_prob > 0:
        mask = torch.rand_like(new_x) < feature_mask_prob
        new_x = new_x.masked_fill(mask, 0.0)

    # Create new Data object instead of deepcopy, preserving all extra attrs
    from torch_geometric.data import Data
    new_data = Data(x=new_x, edge_index=new_edge_index, y=graph.y)
    new_data.num_nodes = int(new_x.size(0))
    # Copy any extra attributes (post_id, etc.) from the original graph
    for key in graph.keys():
        if key not in ('x', 'edge_index', 'y', 'num_nodes'):
            try:
                setattr(new_data, key, getattr(graph, key))
            except Exception:
                pass
    return new_data


class AugmentedGraphDataset:
    def __init__(self, graphs, node_drop_prob=0.0, edge_drop_prob=0.0, feature_mask_prob=0.0, base_seed=42):
        self.graphs = graphs
        self.node_drop_prob = node_drop_prob
        self.edge_drop_prob = edge_drop_prob
        self.feature_mask_prob = feature_mask_prob
        self.base_seed = base_seed

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.node_drop_prob <= 0 and self.edge_drop_prob <= 0 and self.feature_mask_prob <= 0:
            return graph
        return augment_graph(
            graph,
            node_drop_prob=self.node_drop_prob,
            edge_drop_prob=self.edge_drop_prob,
            feature_mask_prob=self.feature_mask_prob,
            seed=self.base_seed + idx,
        )


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
    parser.add_argument('--graph_dir', default='data/processed_graphs_v3')
    parser.add_argument('--metadata', default='data/all_merged_binary.jsonl')

    # Model
    parser.add_argument('--fusion_type', default='cross_attention', choices=['cross_attention', 'gated'])
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--gnn_type', default='gat', choices=['sage', 'gcn', 'gat'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learn_missing_image_emb', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--save_dir', default='models/checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment_node_drop', type=float, default=0.0)
    parser.add_argument('--augment_edge_drop', type=float, default=0.0)
    parser.add_argument('--augment_feature_mask', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing epsilon (0=off)')
    parser.add_argument('--lr_scheduler', default='none', choices=['none', 'cosine', 'cosine_warm'], help='LR scheduler type')
    parser.add_argument('--selection_metric', default='acc', choices=['acc', 'f1'], help='Metric used for checkpoint ranking and early stopping')
    parser.add_argument('--fake_class_weight', type=float, default=2.4, help='Manual class weight for FAKE label')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--ensemble_top_k', type=int, default=5, help='Number of top checkpoints used in ensemble')

    # Ablation
    parser.add_argument('--ablation', default=None, choices=[None, 'text_only', 'image_only', 'no_graph', 'missing_image'])

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Fusion: {args.fusion_type} | GNN: {args.gnn_type} | Ablation: {args.ablation or 'FULL'}")
    logger.info(
        "MissingEmb: %s | Aug(node=%.2f edge=%.2f feat=%.2f)",
        args.learn_missing_image_emb,
        args.augment_node_drop,
        args.augment_edge_drop,
        args.augment_feature_mask,
    )

    # Load data
    train_graphs, val_graphs, test_graphs = load_cascade_graphs(args.graph_dir, args.metadata)
    if not train_graphs:
        logger.error("No training graphs found!")
        return

    # --- USE FULL DATA (No Undersampling) ---
    train_dataset = AugmentedGraphDataset(
        train_graphs,
        node_drop_prob=args.augment_node_drop,
        edge_drop_prob=args.augment_edge_drop,
        feature_mask_prob=args.augment_feature_mask,
        base_seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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

    # Class weights - Balanced for Accuracy (2.4 instead of 2.2 for a bit more FAKE focus)
    full_weights = torch.tensor([1.0, args.fake_class_weight], device=device)
    logger.info(f"Manual Class weights (V5 Optimized): {full_weights.tolist()}")

    # Model
    model = MultimodalCascadeGNN(
        input_dim=input_dim,
        text_dim=text_dim,
        image_dim=image_dim,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.fusion_dim,
        num_classes=args.num_classes,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        fusion_type=args.fusion_type,
        learn_missing_image_emb=args.learn_missing_image_emb,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable:,} trainable")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=full_weights, gamma=args.focal_gamma, smoothing=args.label_smoothing)

    # LR Scheduler
    scheduler = None
    warmup_epochs = 5
    if args.lr_scheduler == 'cosine':
        effective_warmup = min(warmup_epochs, max(1, args.epochs))
        cosine_epochs = max(1, args.epochs - effective_warmup)
        if args.epochs <= effective_warmup:
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
            logger.info(f"LR Scheduler: WarmupOnly({effective_warmup})")
        else:
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=2e-4 * 0.01)
            warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=effective_warmup)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[effective_warmup])
            logger.info(f"LR Scheduler: Warmup({effective_warmup}) + CosineAnnealing({cosine_epochs})")
    elif args.lr_scheduler == 'cosine_warm':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=2e-4 * 0.01)
        logger.info(f"LR Scheduler: CosineAnnealingWarmRestarts (T_0=20)")

    # Tăng patience lên 30 để mô hình có thời gian tìm vùng tối ưu sâu
    selection_key = 'val_acc_bin' if args.selection_metric == 'acc' else 'val_f1_bin'
    best_val_score = float('-inf')
    patience_counter = 0
    patience_limit = 30
    # Top-K Ensemble Setup
    top_k_models = [] # List of (score, path)
    max_k = max(1, int(args.ensemble_top_k))
    
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

        current_score = val_m[selection_key]
        if current_score >= best_val_score:
            best_val_score = current_score
            patience_counter = 0
            # Save top models for ensemble
            model_path = os.path.join(args.save_dir, f'model_{args.selection_metric}_{best_val_score:.4f}_ep{epoch}.pt')
            torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), save_path)
            top_k_models.append((best_val_score, model_path))
            top_k_models = sorted(top_k_models, key=lambda x: x[0], reverse=True)[:max_k]
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()

        if patience_counter >= patience_limit:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # 🎯 TOP-K ENSEMBLE + THRESHOLD TUNING
    if not top_k_models:
        logger.warning("No top-k checkpoints were captured; falling back to the latest canonical checkpoint.")
        if os.path.exists(save_path):
            top_k_models = [(best_val_score, save_path)]
        else:
            torch.save(model.state_dict(), save_path)
            top_k_models = [(best_val_score, save_path)]

    logger.info(f"Performing Ensemble of Top {len(top_k_models)} models...")
    
    def get_ensemble_probs(loader, models_info):
        all_probs = []
        for f1, path in models_info:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            probs_list = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.batch, ablation=args.ablation)
                    probs_list.extend(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            all_probs.append(probs_list)
        return np.mean(all_probs, axis=0)

    val_probs = get_ensemble_probs(val_loader, top_k_models)
    
    # Tuning
    best_threshold = 0.5
    best_val_acc = 0
    val_labels = []
    for batch in val_loader: val_labels.extend(batch.y.numpy())
    
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(val_labels, preds)
        if acc > best_val_acc:
            best_val_acc = acc
            best_threshold = t
    
    logger.info(f"Optimal Ensemble Threshold: {best_threshold:.2f}")

    # Final Test
    test_probs = get_ensemble_probs(test_loader, top_k_models)
    test_labels = []
    for batch in test_loader: test_labels.extend(batch.y.numpy())
    
    test_preds = (test_probs >= best_threshold).astype(int)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='binary')

    ablation_label = (args.ablation or 'FULL').upper()
    print(f"\n{'=' * 50}")
    print(f"FINAL TEST — MULTIMODAL CASCADE GNN ({ablation_label}) ENSEMBLE")
    print(f"Fusion: {args.fusion_type} | Models: {len(top_k_models)} | Threshold: {best_threshold:.2f}")
    print(f"{'=' * 50}")
    print(f"Binary Accuracy:   {test_acc:.4f}")
    print(f"Binary F1:         {test_f1:.4f}")
    print(f"{'=' * 50}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['REAL', 'FAKE'], zero_division=0))

    # Comparison Table
    print(f"\n{'-' * 50}")
    print("VS PHASE 3 BASELINES")
    print(f"{'-' * 50}")
    print(f"{'Model':<30} {'Binary F1':>10} {'Binary Acc':>10}")
    print(f"{'-' * 50}")
    print(f"{'Text-Only (Phase 3)':30} {'0.5250':>10} {'0.5250':>10}")
    print(f"{'Graph-Only GCN (Phase 3)':30} {'0.6320':>10} {'0.6320':>10}")
    print(f"{'Multimodal GNN Ensemble':30} {test_f1:>10.4f} {test_acc:>10.4f}")
    print(f"{'-' * 50}")


if __name__ == '__main__':
    main()
