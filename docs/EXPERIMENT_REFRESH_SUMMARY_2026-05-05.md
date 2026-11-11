# Experiment Refresh Summary (2026-05-05)

This file summarizes the refreshed paper/defense artifacts after migrating from the old 2,384-sample dataset assumptions to the expanded binary dataset:

- Metadata: `data/all_merged_binary.jsonl`
- Graphs: `data/processed_graphs_v3`

## 1. Refreshed Dataset Snapshot

| Metric | Refreshed Value |
|---|---:|
| Total samples | 9,584 |
| Train / Val / Test | 6,709 / 1,438 / 1,437 |
| REAL / FAKE | 7,576 / 2,008 |
| Cascade graphs (`num_nodes > 1`) | 7,916 / 9,584 (`82.60%`) |
| Single-node graphs | 1,668 / 9,584 (`17.40%`) |
| Feature dimension (v3 multimodal) | 1,793 |
| Test REAL single-node | 146 / 290 (`50.34%`) |
| Test FAKE single-node | 44 / 117 (`37.61%`) |

Key implication:

- The old assumption "roughly 60% of the dataset is single-node" is no longer valid.
- Any narrative built on the old `2,384`-sample split should be considered stale and superseded by this refresh.

## 2. Refreshed Ablation Metrics

These numbers were rerun on the refreshed split with a short, consistent evaluation setup to replace the stale low-60% figures from the old plan.

| Config | Binary Acc | Binary F1 |
|---|---:|---:|
| Graph GAT | 83.29% | 74.24% |
| Full Multimodal GNN (CrossAtt) | 87.47% | 79.18% |
| Text-only ablation | 78.62% | 63.60% |
| Image-only ablation | 87.47% | 79.84% |
| No-Graph | 88.45% | 80.66% |
| Gated Fusion | 86.49% | 78.76% |

Interpretation:

- Full Cross-Attention remains better than Graph-only by `+4.18 pp` accuracy.
- Cross-Attention remains better than Gated Fusion by `+0.98 pp`.
- On the refreshed split, `No-Graph` slightly outperforms the full graph path by `+0.98 pp`, so the contribution of cascade enrichment must now be argued carefully and validated through TN5 rather than assumed from older numbers.

## 3. Refreshed Experiment Outputs

### TN1 - Ablation Package

Generated from:

- `reports/results/paper_experiments/ablation_refresh_20260505.json`

Updated outputs:

- `reports/results/paper_figures/ablation_definitive.png`
- `reports/results/paper_figures/ablation_definitive.tex`
- `reports/results/paper_figures/ablation_definitive_summary.md`

### TN2 - Error Analysis

Refreshed results:

- Overall accuracy: `87.96%`
- REAL precision / recall: `91.99% / 91.03%`
- FAKE precision / recall: `78.33% / 80.34%`
- Accuracy by graph size:
  - `0 comments`: `87.37%` on `190` samples
  - `1-10 comments`: `83.93%` on `112` samples
  - `>10 comments`: `93.33%` on `105` samples
- Confidence reliability:
  - `medium`: `69.85%` accuracy, avg confidence `61.66%`
  - `high`: `97.05%` accuracy, avg confidence `83.74%`

Updated outputs:

- `reports/results/error_analysis_report.json`
- `reports/results/error_analysis_report.md`
- `reports/results/paper_figures/confusion_matrix.png`
- `reports/results/paper_figures/accuracy_by_graph_size.png`
- `reports/results/paper_figures/confidence_reliability.png`

### TN3 - Robustness

Refreshed results:

- Missing image:
  - Full accuracy: `87.96%`
  - Missing-image accuracy: `71.25%`
  - Delta: `-16.71 pp`
- Sparse cascade (`comment_count > 5` subset):
  - Full / Root+1 / Root+3 all stayed at `93.98%`
  - Delta: `0.00 pp`
- Adversarial comment injection:
  - Sample size: `50`
  - Injected comments per sample: `5`
  - Flip rate: `80.00%`

Updated outputs:

- `reports/results/robustness_report.json`
- `reports/results/paper_figures/robustness_missing_image.png`
- `reports/results/paper_figures/robustness_sparse_cascade.png`
- `reports/results/paper_figures/robustness_adversarial.png`

### TN5 - Cascade Contribution

Refreshed results:

- With cascade accuracy / F1: `87.96% / 79.32%`
- Root-only accuracy / F1: `87.96% / 79.32%`
- Delta: `0.00 pp`

By comment bucket:

- `0`: `87.37%` vs `87.37%`
- `1-5`: `79.76%` vs `79.76%`
- `6-20`: `94.96%` vs `94.96%`
- `>20`: `85.71%` vs `85.71%`

Topology differences:

- Depth Mann-Whitney `p = 3.72e-05`
- Average degree Mann-Whitney `p = 3.22e-05`
- Branching factor Mann-Whitney `p = 0.2812`

Interpretation:

- On the refreshed checkpoint/split combination used here, the root-only and full-cascade predictions are identical, which materially weakens the old "cascade always adds accuracy" narrative.
- The topology statistics still show structural differences between REAL and FAKE, especially on graph depth and average degree.

Updated outputs:

- `reports/results/cascade_contribution_report.json`
- `reports/results/paper_figures/cascade_vs_no_cascade.png`
- `reports/results/paper_figures/accuracy_vs_cascade_size.png`
- `reports/results/paper_figures/topology_real_vs_fake.png`

## 4. Code Changes Applied

- Updated `scripts/paper_experiments/common.py` to default to `data/all_merged_binary.jsonl`.
- Split compute vs render for paper experiments so the broken `matplotlib` inside `multimodal_gnn` no longer blocks artifact refresh:
  - `scripts/paper_experiments/02_error_analysis.py`
  - `scripts/paper_experiments/03_robustness_test.py`
  - `scripts/paper_experiments/05_cascade_contribution.py`
- Added `scripts/paper_experiments/render_refreshed_binary_figures.py` to render refreshed images with the system Python that has a working matplotlib installation.
- Added refreshed ablation input file:
  - `reports/results/paper_experiments/ablation_refresh_20260505.json`
- Updated `scripts/paper_experiments/01_ablation_table.py` summary/LaTeX phrasing so negative `Full vs No-Graph` deltas are reported honestly.

## 5. Important Caveats

- `docs/WINNING_EXPERIMENT_PLAN.md` still contains stale top-level numbers and also has mojibake/encoding issues; use this refresh summary as the authoritative replacement until that document is manually cleaned and rewritten.
- `scripts/paper_experiments/04_domain_shift.py` was not rerun in this refresh because it depends on live Reddit fetching and online inference conditions; the current environment is not suitable for that live refresh.
- The refreshed figures in `reports/results/paper_figures/` should be treated as the new paper/slide artifacts for TN1, TN2, TN3, and TN5.
