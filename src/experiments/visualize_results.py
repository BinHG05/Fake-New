"""
Generate Paper Visualizations.

Creates confusion matrices, comparison bar charts, and ablation plots
for the research paper.

Usage:
    # Generate all visualizations from results JSON
    python src/experiments/visualize_results.py

    # Generate from a specific results file
    python src/experiments/visualize_results.py --results results/paper_experiments/all_results_*.json

    # Generate confusion matrix from a single model run
    python src/experiments/visualize_results.py --mode confusion_matrix \
        --graph_dir data/processed_graphs_multimodal --checkpoint models/checkpoints/best_multimodal_cascade_cross_attention.pt
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_MAP = {
    'TRUE': 0, 'MOSTLY_TRUE': 1, 'HALF_TRUE': 2,
    'BARELY_TRUE': 3, 'FALSE': 4, 'PANTS_ON_FIRE': 5
}
LABEL_NAMES = ['TRUE', 'MOSTLY_TRUE', 'HALF_TRUE', 'BARELY_TRUE', 'FALSE', 'PANTS_ON_FIRE']
SHORT_NAMES = ['TRUE', 'M_TRUE', 'H_TRUE', 'B_TRUE', 'FALSE', 'PANTS']

OUTPUT_DIR = 'results/paper_figures'

DISPLAY_NAME_MAP = {
    'text_only': 'Text Only',
    'image_only': 'Image Only',
    'fusion': 'Early Fusion',
    'graph_gcn': 'Graph GCN',
    'graph_sage': 'Graph SAGE',
    'graph_gat': 'Graph GAT',
    'full_model': 'Full Model',
    'no_graph': 'No Graph',
    'text_graph_only': 'Text Graph Only',
    'gated_fusion': 'Gated Fusion',
}


def prettify_model_name(name):
    return DISPLAY_NAME_MAP.get(name, name.replace('_', ' ').title())


def to_percent(value):
    return float(value or 0) * 100.0


def load_test_data(graph_dir, metadata_path):
    """Load test graphs."""
    id_to_meta = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if 'label' not in item or item['label'] not in LABEL_MAP:
                continue
            id_to_meta[item['id']] = {
                'label': LABEL_MAP[item['label']],
                'split': item['split']
            }

    torch.serialization.add_safe_globals([Data])
    test_graphs = []
    for pt_file in sorted(Path(graph_dir).glob('*.pt')):
        post_id = pt_file.stem
        if post_id not in id_to_meta:
            continue
        if id_to_meta[post_id]['split'] != 'test':
            continue
        data = torch.load(pt_file, weights_only=False)
        data.y = torch.tensor([id_to_meta[post_id]['label']], dtype=torch.long)
        if data.edge_index.numel() == 0:
            data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        test_graphs.append(data)

    return test_graphs


def get_predictions(model, test_loader, device, ablation=None):
    """Get predictions from model."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch, ablation=ablation)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """Generate and save confusion matrix plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=range(6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                ax=axes[0])
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                ax=axes[1])
    axes[1].set_title(f'{title} (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_comparison_chart(results_dict, save_path=None):
    """Generate bar chart comparing all models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = list(results_dict.keys())
    f1_scores = [results_dict[m].get('f1_6', 0) for m in models]
    bin_accs = [results_dict[m].get('acc_bin', 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, f1_scores, width, label='6-Class Macro-F1',
                   color='#4A90D9', edgecolor='white')
    bars2 = ax.bar(x + width/2, bin_accs, width, label='Binary Accuracy',
                   color='#E8744F', edgecolor='white')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_binary_percentage_chart(results_dict, save_path=None):
    """Plot Binary Accuracy and Binary F1 in percentage for each model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = list(results_dict.keys())
    display_names = [prettify_model_name(m) for m in models]
    acc_values = [to_percent(results_dict[m].get('acc_bin', 0)) for m in models]
    f1_values = [to_percent(results_dict[m].get('f1_bin', 0)) for m in models]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(15, 7))
    bars1 = ax.bar(x - width / 2, acc_values, width, label='Binary Accuracy (%)',
                   color='#1f77b4', edgecolor='white')
    bars2 = ax.bar(x + width / 2, f1_values, width, label='Binary F1 (%)',
                   color='#ff7f0e', edgecolor='white')

    ax.set_xlabel('Model / Technology', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Binary Performance by Model', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=35, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(acc_values + f1_values + [100]) * 1.05 if (acc_values or f1_values) else 100)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_delta_vs_reference(results_dict, reference_key, metric='acc_bin', title=None, save_path=None):
    """Plot percentage-point gain/loss against a reference model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if reference_key not in results_dict:
        logger.warning(f"Reference key not found for delta plot: {reference_key}")
        return

    reference_value = to_percent(results_dict[reference_key].get(metric, 0))
    labels = []
    deltas = []

    for key, metrics in results_dict.items():
        if key == reference_key:
            continue
        labels.append(prettify_model_name(key))
        deltas.append(to_percent(metrics.get(metric, 0)) - reference_value)

    if not labels:
        return

    colors = ['#2ca02c' if delta >= 0 else '#d62728' for delta in deltas]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(labels, deltas, color=colors, edgecolor='white')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Delta (percentage points)', fontsize=12)
    ax.set_title(title or f'Delta vs {prettify_model_name(reference_key)}', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')

    for bar, delta in zip(bars, deltas):
        ax.annotate(f'{delta:+.2f} pp',
                    xy=(bar.get_x() + bar.get_width() / 2, delta),
                    xytext=(0, 3 if delta >= 0 else -15),
                    textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_ablation_chart(ablation_results, save_path=None):
    """Generate ablation study chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = list(ablation_results.keys())
    f1_scores = [ablation_results[m].get('f1_6', 0) for m in models]

    # Color code: full model = green, ablations = shades
    colors = []
    for m in models:
        if m == 'full_model':
            colors.append('#2ECC71')
        elif 'gated' in m:
            colors.append('#9B59B6')
        else:
            colors.append('#E74C3C')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, f1_scores, color=colors, edgecolor='white', height=0.6)

    ax.set_xlabel('6-Class Macro-F1', fontsize=12)
    ax.set_title('Ablation Study: Component Contribution', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_ablation_binary_acc_chart(ablation_results, save_path=None):
    """Generate ablation study chart for Binary Accuracy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = list(ablation_results.keys())
    bin_accs = [to_percent(ablation_results[m].get('acc_bin', 0)) for m in models]

    # Color code: full model = green, ablations = shades
    colors = []
    for m in models:
        if m == 'full_model':
            colors.append('#2ECC71')
        elif 'gated' in m:
            colors.append('#9B59B6')
        else:
            colors.append('#E74C3C')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, bin_accs, color=colors, edgecolor='white', height=0.6)

    ax.set_xlabel('Binary Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study: Binary Accuracy Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, bin_accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def plot_multi_seed_boxplot(multi_seed_results, save_path=None):
    """Generate box plot for multi-seed runs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if 'summary' not in multi_seed_results:
        return

    summary = multi_seed_results['summary']
    metrics = ['f1_6', 'acc_6', 'acc_bin', 'f1_bin']
    labels = ['6-Class F1 (%)', '6-Class Acc (%)', 'Binary Acc (%)', 'Binary F1 (%)']

    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    valid_labels = []
    for metric, label in zip(metrics, labels):
        if metric in summary and 'values' in summary[metric]:
            data.append([to_percent(v) for v in summary[metric]['values']])
            valid_labels.append(label)

    if data:
        bp = ax.boxplot(data, tick_labels=valid_labels, patch_artist=True)
        colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Multi-Seed Results Distribution (n=5)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean ± std annotations
        for i, (metric, label) in enumerate(zip(metrics, valid_labels)):
            if metric in summary:
                mean = summary[metric]['mean']
                std = summary[metric]['std']
                ax.text(i + 1, mean + 0.02, f'{mean:.4f}±{std:.4f}',
                        ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    plt.close()


def generate_binary_summary_markdown(all_results, save_path=None):
    """Generate a concise markdown summary focused on percentage and delta."""
    lines = []
    lines.append('# Binary Results Summary')
    lines.append('')

    baselines = all_results.get('baselines', {})
    ablation = all_results.get('ablation', {})

    if baselines:
        best_baseline_key = max(baselines, key=lambda k: baselines[k].get('acc_bin', 0))
        best_baseline = baselines[best_baseline_key]
        lines.append(f"- Best baseline: **{prettify_model_name(best_baseline_key)}**")
        lines.append(f"  Binary Accuracy: **{to_percent(best_baseline.get('acc_bin', 0)):.2f}%**")
        lines.append(f"  Binary F1: **{to_percent(best_baseline.get('f1_bin', 0)):.2f}%**")
        lines.append('')

    if ablation and baselines:
        if 'full_model' in ablation:
            full_model = ablation['full_model']
            baseline_acc = to_percent(best_baseline.get('acc_bin', 0))
            baseline_f1 = to_percent(best_baseline.get('f1_bin', 0))
            full_acc = to_percent(full_model.get('acc_bin', 0))
            full_f1 = to_percent(full_model.get('f1_bin', 0))

            lines.append(f"- Full model: **{full_acc:.2f}%** Binary Accuracy, **{full_f1:.2f}%** Binary F1")
            lines.append(f"- Delta vs best baseline:")
            lines.append(f"  Accuracy: **{full_acc - baseline_acc:+.2f} pp**")
            lines.append(f"  F1: **{full_f1 - baseline_f1:+.2f} pp**")
            lines.append('')

    if ablation and 'full_model' in ablation:
        full_acc = to_percent(ablation['full_model'].get('acc_bin', 0))
        lines.append('- Ablation delta vs Full Model:')
        for key, metrics in ablation.items():
            if key == 'full_model':
                continue
            acc = to_percent(metrics.get('acc_bin', 0))
            lines.append(f"  - {prettify_model_name(key)}: {acc:.2f}% ({acc - full_acc:+.2f} pp)")

    content = '\n'.join(lines) + '\n'
    print(content)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved: {save_path}")


def generate_latex_table(all_results, save_path=None):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Comprehensive Model Comparison}')
    lines.append(r'\label{tab:results}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\hline')
    lines.append(r'\textbf{Model} & \textbf{6-Class F1} & \textbf{6-Class Acc} & \textbf{Binary Acc} \\')
    lines.append(r'\hline')

    if 'baselines' in all_results:
        lines.append(r'\multicolumn{4}{l}{\textit{Phase 3 Baselines}} \\')
        for name, m in all_results['baselines'].items():
            if m:
                lines.append(f"  {name.replace('_', ' ').title()} & "
                             f"{m.get('f1_6', 0):.4f} & "
                             f"{m.get('acc_6', 0):.4f} & "
                             f"{m.get('acc_bin', 0):.4f} \\\\")
        lines.append(r'\hline')

    if 'ablation' in all_results:
        lines.append(r'\multicolumn{4}{l}{\textit{Phase 4 Ablation}} \\')
        for name, m in all_results['ablation'].items():
            if m:
                bold = r'\textbf' if name == 'full_model' else ''
                lines.append(f"  {bold}{{{name.replace('_', ' ').title()}}} & "
                             f"{m.get('f1_6', 0):.4f} & "
                             f"{m.get('acc_6', 0):.4f} & "
                             f"{m.get('acc_bin', 0):.4f} \\\\")

    if 'multi_seed' in all_results and 'summary' in all_results['multi_seed']:
        s = all_results['multi_seed']['summary']
        lines.append(r'\hline')
        f1 = s.get('f1_6', {})
        acc6 = s.get('acc_6', {})
        accb = s.get('acc_bin', {})
        lines.append(f"  \\textbf{{Ours (mean $\\pm$ std)}} & "
                     f"${f1.get('mean', 0):.4f} \\pm {f1.get('std', 0):.4f}$ & "
                     f"${acc6.get('mean', 0):.4f} \\pm {acc6.get('std', 0):.4f}$ & "
                     f"${accb.get('mean', 0):.4f} \\pm {accb.get('std', 0):.4f}$ \\\\")

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    table = '\n'.join(lines)
    print(table)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table)
        logger.info(f"LaTeX table saved to {save_path}")


def generate_confusion_from_checkpoint(graph_dir, checkpoint_path, metadata_path):
    """Load model from checkpoint and generate confusion matrix."""
    from src.training.train_multimodal_gnn import MultimodalCascadeGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_graphs = load_test_data(graph_dir, metadata_path)

    if not test_graphs:
        logger.error("No test graphs found!")
        return

    test_loader = DataLoader(test_graphs, batch_size=32)
    input_dim = test_graphs[0].x.size(1)

    if input_dim > 768:
        text_dim, image_dim = 768, input_dim - 768
    else:
        text_dim, image_dim = input_dim, 0

    model = MultimodalCascadeGNN(
        input_dim=input_dim, text_dim=text_dim, image_dim=image_dim,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    preds, targets = get_predictions(model, test_loader, device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(
        targets, preds,
        title='Multimodal GNN',
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix_multimodal_gnn.png')
    )

    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(targets, preds,
                                target_names=LABEL_NAMES,
                                labels=range(6), zero_division=0))


def main():
    parser = argparse.ArgumentParser(description='Generate paper visualizations')
    parser.add_argument('--mode', default='from_results',
                        choices=['from_results', 'confusion_matrix'])
    parser.add_argument('--results', default=None, help='Path to results JSON')
    parser.add_argument('--graph_dir', default='data/processed_graphs_multimodal')
    parser.add_argument('--checkpoint', default='models/checkpoints/best_multimodal_cascade_cross_attention.pt')
    parser.add_argument('--metadata', default='data/reddit_enriched_data.jsonl')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.mode == 'confusion_matrix':
        generate_confusion_from_checkpoint(args.graph_dir, args.checkpoint, args.metadata)
        return

    # Load results
    if args.results:
        with open(args.results) as f:
            all_results = json.load(f)
    else:
        # Try to find latest results file
        results_dir = Path('results/paper_experiments')
        if results_dir.exists():
            files = sorted(results_dir.glob('all_results_*.json'))
            if files:
                with open(files[-1]) as f:
                    all_results = json.load(f)
                logger.info(f"Loaded: {files[-1]}")
            else:
                logger.error("No results files found. Run run_paper_experiments.py first!")
                return
        else:
            logger.error("No results directory found!")
            return

    # Generate all plots
    if 'baselines' in all_results and 'ablation' in all_results:
        combined = {}
        combined.update(all_results.get('baselines', {}))
        combined.update(all_results.get('ablation', {}))
        plot_comparison_chart(combined,
                              save_path=os.path.join(OUTPUT_DIR, 'model_comparison.png'))
        plot_binary_percentage_chart(
            combined,
            save_path=os.path.join(OUTPUT_DIR, 'binary_percentage_comparison.png')
        )

        best_baseline_key = max(
            all_results['baselines'],
            key=lambda k: all_results['baselines'][k].get('acc_bin', 0)
        )
        plot_delta_vs_reference(
            combined,
            reference_key=best_baseline_key,
            metric='acc_bin',
            title=f'Binary Accuracy Delta vs {prettify_model_name(best_baseline_key)}',
            save_path=os.path.join(OUTPUT_DIR, 'binary_accuracy_delta_vs_best_baseline.png')
        )
        plot_delta_vs_reference(
            combined,
            reference_key=best_baseline_key,
            metric='f1_bin',
            title=f'Binary F1 Delta vs {prettify_model_name(best_baseline_key)}',
            save_path=os.path.join(OUTPUT_DIR, 'binary_f1_delta_vs_best_baseline.png')
        )

    if 'ablation' in all_results:
        plot_ablation_chart(all_results['ablation'],
                            save_path=os.path.join(OUTPUT_DIR, 'ablation_study.png'))
        plot_ablation_binary_acc_chart(all_results['ablation'],
                                       save_path=os.path.join(OUTPUT_DIR, 'ablation_study_binary.png'))
        if 'full_model' in all_results['ablation']:
            plot_delta_vs_reference(
                all_results['ablation'],
                reference_key='full_model',
                metric='acc_bin',
                title='Binary Accuracy Delta vs Full Model',
                save_path=os.path.join(OUTPUT_DIR, 'ablation_binary_delta_vs_full_model.png')
            )
            plot_delta_vs_reference(
                all_results['ablation'],
                reference_key='full_model',
                metric='f1_bin',
                title='Binary F1 Delta vs Full Model',
                save_path=os.path.join(OUTPUT_DIR, 'ablation_binary_f1_delta_vs_full_model.png')
            )

    if 'multi_seed' in all_results:
        plot_multi_seed_boxplot(all_results['multi_seed'],
                                save_path=os.path.join(OUTPUT_DIR, 'multi_seed_boxplot.png'))

    # Generate LaTeX table
    generate_latex_table(all_results,
                         save_path=os.path.join(OUTPUT_DIR, 'results_table.tex'))
    generate_binary_summary_markdown(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, 'binary_results_summary.md')
    )

    logger.info(f"All figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
