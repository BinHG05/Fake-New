"""
Run All Experiments for Paper.

Runs ablation studies, multi-seed experiments, and generates
comprehensive results for the research paper.

Usage:
    # Run everything (ablation + multi-seed)
    python src/experiments/run_paper_experiments.py

    # Only ablation study
    python src/experiments/run_paper_experiments.py --mode ablation

    # Only multi-seed runs
    python src/experiments/run_paper_experiments.py --mode multi_seed

    # Only baselines (Phase 3)
    python src/experiments/run_paper_experiments.py --mode baselines
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = 'results/paper_experiments'


def run_command(cmd, description):
    """Run a training command and capture output."""
    logger.info(f"{'='*60}")
    logger.info(f"  {description}")
    logger.info(f"CMD: {cmd}")
    logger.info(f"{'='*60}")

    # Use bytes mode to avoid Windows Unicode issues with emoji in logs
    result = subprocess.run(
        cmd, shell=True, capture_output=True
    )

    stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
    stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
    output = stdout + stderr
    print(output)

    # Parse results from output
    metrics = parse_metrics(output)
    return metrics, output


def parse_metrics(output):
    """Parse test metrics from training output."""
    metrics = {}
    for line in output.split('\n'):
        line = line.strip()
        if '6-Class Accuracy:' in line:
            metrics['acc_6'] = float(line.split(':')[-1].strip())
        elif '6-Class Macro-F1:' in line:
            metrics['f1_6'] = float(line.split(':')[-1].strip())
        elif 'Binary Accuracy:' in line:
            metrics['acc_bin'] = float(line.split(':')[-1].strip())
        elif 'Binary F1:' in line:
            metrics['f1_bin'] = float(line.split(':')[-1].strip())
    return metrics


def run_baselines():
    """Run Phase 3 baselines."""
    results = {}

    experiments = [
        ('text_only', 'python src/training/train_baseline.py --model_type text --epochs 20 --lr 2e-4'),
        ('image_only', 'python src/training/train_baseline.py --model_type image --epochs 20 --lr 2e-4'),
        ('fusion', 'python src/training/train_baseline.py --model_type fusion --epochs 20 --lr 2e-4'),
        ('graph_gcn', 'python src/training/train_gnn.py --epochs 100 --gnn_type gcn --patience 15'),
        ('graph_sage', 'python src/training/train_gnn.py --epochs 100 --gnn_type sage --patience 15'),
        ('graph_gat', 'python src/training/train_gnn.py --epochs 100 --gnn_type gat --patience 15'),
    ]

    for name, cmd in experiments:
        metrics, output = run_command(cmd, f"Baseline: {name}")
        results[name] = metrics

    return results


def run_ablation(graph_dir='data/processed_graphs_multimodal'):
    """Run ablation study on Phase 4 model."""
    results = {}

    experiments = [
        ('full_model', f'python src/training/train_multimodal_gnn.py --graph_dir {graph_dir} --epochs 100 --lr 1e-3 --patience 15 --seed 42'),
        ('text_only', f'python src/training/train_multimodal_gnn.py --graph_dir {graph_dir} --epochs 100 --lr 1e-3 --patience 15 --ablation text_only --seed 42'),
        ('image_only', f'python src/training/train_multimodal_gnn.py --graph_dir {graph_dir} --epochs 100 --lr 1e-3 --patience 15 --ablation image_only --seed 42'),
        ('no_graph', f'python src/training/train_multimodal_gnn.py --graph_dir {graph_dir} --epochs 100 --lr 1e-3 --patience 15 --ablation no_graph --seed 42'),
    ]

    # Also test with text-only graphs (no image features)
    experiments.append(
        ('text_graph_only', 'python src/training/train_multimodal_gnn.py --graph_dir data/processed_graphs --epochs 100 --lr 1e-3 --patience 15 --seed 42'),
    )

    # Test different fusion types
    experiments.append(
        ('gated_fusion', f'python src/training/train_multimodal_gnn.py --graph_dir {graph_dir} --epochs 100 --lr 1e-3 --patience 15 --fusion_type gated --seed 42'),
    )

    for name, cmd in experiments:
        metrics, output = run_command(cmd, f"Ablation: {name}")
        results[name] = metrics

    return results


def run_multi_seed(graph_dir='data/processed_graphs_multimodal', n_seeds=5):
    """Run full model with multiple seeds for statistical significance."""
    import numpy as np

    seeds = [42, 123, 456, 789, 1024][:n_seeds]
    all_metrics = []

    for seed in seeds:
        cmd = (f'python src/training/train_multimodal_gnn.py '
               f'--graph_dir {graph_dir} --epochs 100 --lr 1e-3 --patience 15 --seed {seed}')
        metrics, output = run_command(cmd, f"Multi-seed run (seed={seed})")
        if metrics:
            all_metrics.append(metrics)

    # Compute mean ± std
    if all_metrics:
        summary = {}
        for key in ['f1_6', 'acc_6', 'acc_bin', 'f1_bin']:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': values
                }

        return {'individual': all_metrics, 'summary': summary}
    return {}


def save_results(results, filename):
    """Save results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {path}")


def print_summary_table(all_results):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'6-Class F1':>10} {'6-Class Acc':>11} {'Binary Acc':>10}")
    print(f"{'─'*70}")

    if 'baselines' in all_results:
        print("Phase 3 Baselines:")
        for name, m in all_results['baselines'].items():
            if m:
                print(f"  {name:<33} {m.get('f1_6',0):>10.4f} {m.get('acc_6',0):>11.4f} {m.get('acc_bin',0):>10.4f}")

    if 'ablation' in all_results:
        print(f"\nPhase 4 Ablation Study:")
        for name, m in all_results['ablation'].items():
            if m:
                print(f"  {name:<33} {m.get('f1_6',0):>10.4f} {m.get('acc_6',0):>11.4f} {m.get('acc_bin',0):>10.4f}")

    if 'multi_seed' in all_results and 'summary' in all_results['multi_seed']:
        s = all_results['multi_seed']['summary']
        print(f"\nPhase 4 Multi-Seed (mean ± std):")
        f1 = s.get('f1_6', {})
        acc6 = s.get('acc_6', {})
        accb = s.get('acc_bin', {})
        print(f"  {'Multimodal GNN':<33} "
              f"{f1.get('mean',0):>.4f}±{f1.get('std',0):.4f} "
              f"{acc6.get('mean',0):>.4f}±{acc6.get('std',0):.4f} "
              f"{accb.get('mean',0):>.4f}±{accb.get('std',0):.4f}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Run all paper experiments')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'baselines', 'ablation', 'multi_seed'],
                        help='Which experiments to run')
    parser.add_argument('--graph_dir', default='data/processed_graphs_multimodal')
    parser.add_argument('--n_seeds', type=int, default=5)
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = {}

    if args.mode in ('all', 'baselines'):
        logger.info("Phase 3: Running Baselines...")
        all_results['baselines'] = run_baselines()
        save_results(all_results['baselines'], f'baselines_{timestamp}.json')

    if args.mode in ('all', 'ablation'):
        logger.info("Phase 4: Running Ablation Study...")
        all_results['ablation'] = run_ablation(args.graph_dir)
        save_results(all_results['ablation'], f'ablation_{timestamp}.json')

    if args.mode in ('all', 'multi_seed'):
        logger.info("Phase 4: Running Multi-Seed Experiments...")
        all_results['multi_seed'] = run_multi_seed(args.graph_dir, args.n_seeds)
        save_results(all_results['multi_seed'], f'multi_seed_{timestamp}.json')

    # Save combined results
    save_results(all_results, f'all_results_{timestamp}.json')

    # Print summary
    print_summary_table(all_results)


if __name__ == '__main__':
    main()
