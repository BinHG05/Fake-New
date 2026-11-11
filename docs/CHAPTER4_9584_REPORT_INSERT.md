# Chapter 4 Update for the 9,584-Sample Dataset

This file reflects the **new rerun-backed Chapter 4 numbers** for the
`9,584`-sample dataset.

The key change is that the refreshed training pipeline has been rerun with a
more stable full-model setup, so the chapter no longer has to rely on the old
`No-Graph > Full` artifact.

Primary sources:

- `reports/results/paper_experiments/consistent_rerun_summary_20260505.json`
- `reports/results/paper_experiments/multi_seed_refresh_20260505.json`

---

## 1. Canonical Numbers to Use

Use this checkpoint as the **main Full Model result for Chapter 4**:

- Full model: `87.71%` Binary Accuracy
- Binary F1: `79.17%`
- Model: Multimodal Cascade GNN + Cross-Attention + GAT
- Seed: `42`

This immediately fixes the previous inconsistency:

- `Full Model = 87.71%`
- `No-Graph ablation = 86.49%`

So in the new rerun bundle:

- **Full Model > No-Graph**

by:

- `+1.22 pp`

---

## 2. Section 4.1.1 - Dataset

Use these dataset statistics:

- Total cascades: `9,584`
- Train / Validation / Test: `6,709 / 1,438 / 1,437`
- REAL / FAKE: `7,576 / 2,008`
- REAL / FAKE ratio: `79.05% / 20.95%`
- Image coverage: `9,389 / 9,584 = 97.97%`

### Suggested paragraph

After preprocessing and propagation-graph recovery, the refreshed dataset
contains **9,584** cascades. The split includes **6,709** training samples,
**1,438** validation samples, and **1,437** test samples. The label distribution is
imbalanced, with **7,576 REAL** samples and **2,008 FAKE** samples, equivalent
to **79.05%** and **20.95%**, respectively. Visual coverage is high, with image
information available for **97.97%** of the collected posts.

### Table 4.1

| Dataset split | Number of cascades | Ratio (%) |
|---|---:|---:|
| Train | 6,709 | 70.00 |
| Validation | 1,438 | 15.00 |
| Test | 1,437 | 15.00 |
| Total | 9,584 | 100.00 |

### Table 4.2

| Label | Count | Percentage |
|---|---:|---:|
| REAL | 7,576 | 79.05 |
| FAKE | 2,008 | 20.95 |
| Total | 9,584 | 100.00 |

---

## 3. Section 4.2 - Baselines

### Important naming rule

For the report:

- `Late Fusion` = **Text + Image only, no GNN**
- `No-Graph` = ablation inside the multimodal graph model

Do not use these two names interchangeably.

### Ready-to-use baseline numbers

| Baseline model | Features used | Binary Accuracy (%) |
|---|---|---:|
| Text-Only Baseline | Text only | 78.62 |
| Image-Only Baseline | Image only | 87.47 |
| Late Fusion | Text + Image, no GNN | 72.24 |
| Graph-Only Baseline (GAT) | Text + propagation graph | 83.29 |

### Suggested paragraph

On the refreshed 9,584-sample dataset, the graph-only baseline reaches
**83.29%** accuracy, outperforming the text-only baseline at **78.62%**. This
confirms that propagation structure still contributes useful information. The
strict Late Fusion baseline, which uses only text and image without graph
propagation, achieves **72.24%**, remaining clearly below the graph-aware
models. These results support the importance of the propagation component rather
than reducing the task to content-only fusion.

---

## 4. Section 4.3 - Overall Result

### Canonical full-model result

- **Binary Accuracy = 87.71%**
- **Binary F1 = 79.17%**

### Suggested paragraph

On the refreshed 9,584-sample binary dataset, the full multimodal cascade GNN
with cross-attention fusion achieves **87.71%** binary accuracy and **79.17%**
binary F1-score. Compared with the graph-only baseline (**83.29%**), the full
model improves accuracy by **4.42 percentage points**, showing that the best
performance is achieved when textual, visual, and propagation cues are modeled
jointly.

---

## 5. Section 4.4 - Ablation Study

Use this rerun-backed table:

| Experimental variant | Binary Accuracy (%) | Delta vs Full (pp) |
|---|---:|---:|
| Full Model (Cascade GNN, Cross-Attention) | 87.71 | 0.00 |
| Missing-image variant | 81.08 | -6.63 |
| No-Graph ablation | 86.49 | -1.22 |
| Gated Fusion | 87.71 | 0.00 |

### Additional supporting result

- Graph-aware text-only ablation: `82.80%`

This can be used in the discussion to show that:

- propagation + text alone is useful
- adding image still improves the final model

### Suggested paragraph

The rerun ablation results now show that removing graph propagation reduces
accuracy from **87.71%** to **86.49%**, confirming that the GNN branch makes a
positive contribution in the refreshed setting. When image information is
suppressed, performance drops more sharply to **81.08%**, indicating that visual
features are also an important component of the multimodal architecture.
Together, these results support the thesis that the strongest model is obtained
by combining graph propagation, text, and image information rather than relying
on any single modality alone.

### Practical note

On this rerun, `Gated Fusion` and `Cross-Attention` are effectively tied in
accuracy (`87.71%` vs `87.71%`), so do not overclaim a strong gap between the
two in the refreshed chapter unless you rerun a larger comparison.

---

## 6. Section 4.5 - Propagation Structure Analysis

Use these refreshed topology values:

| Graph property | REAL | FAKE |
|---|---:|---:|
| Mean depth | 0.76 | 1.56 |
| Branching factor | 5.30 | 1.15 |

### Statistical test values

- Depth p-value: `3.72e-05`
- Branching-factor p-value: `0.2812`

### Suggested paragraph

The refreshed topology analysis confirms that FAKE cascades remain deeper than
REAL cascades, with mean depths of **1.56** and **0.76**, respectively. This
depth difference is statistically significant (`p = 3.72e-05`), indicating that
propagation depth still provides useful structural evidence for graph-based fake
news detection on the expanded dataset.

Important note:

- the branching-factor difference is not significant on the refreshed split
- so the old claim `p < 1e-8` should not be reused

---

## 7. Section 4.7.1 - Multi-seed Robustness

This section can now be filled with real rerun-backed numbers.

### Per-seed results

| Random seed | Binary Accuracy (%) | Binary F1-Score (%) |
|---|---:|---:|
| 42 | 87.71 | 79.67 |
| 123 | 87.22 | 78.51 |
| 456 | 86.24 | 77.24 |
| 789 | 88.21 | 80.65 |
| 1024 | 88.21 | 79.49 |
| Mean +/- Std | 87.52 +/- 0.74 | 79.11 +/- 1.16 |

### Suggested paragraph

To evaluate training stability, the refreshed full model was rerun with five
different random seeds. The model achieves a mean binary accuracy of
**87.52% +/- 0.74%** and a mean binary F1-score of **79.11% +/- 1.16%**. The
relatively small standard deviation indicates that the refreshed training setup
is reasonably stable across repeated runs.

---

## 8. Section 4.8.1 - Confusion Matrix

The currently available confusion matrix in the repo still comes from:

- `reports/results/error_analysis_report.json`

with:

- TN = `264`
- FP = `26`
- FN = `23`
- TP = `94`

If you keep this figure, label it as:

- `Refreshed error-analysis checkpoint`

and do not present it as if it was produced by the new canonical rerun above.

---

## 9. Section 4.9.3 - Illustrative Attention Case

Use:

- top comment attention score = `0.50`

### Suggested paragraph

In the refreshed explainability case study, the most influential comment node
receives an attention score of **0.50**, indicating that a single node can
contribute approximately half of the dominant local attention mass in the final
decision for that example.

---

## 10. Final Position for Chapter 4

If you want the chapter to stay both truthful and aligned with the graph
propagation focus, the cleanest message is:

1. The dataset is now much larger, with `9,584` cascades.
2. Graph-aware modeling still matters because `Graph-only = 83.29%` is above `Text-only = 78.62%`.
3. Strict `Late Fusion = 72.24%`, so content-only fusion without GNN is not enough.
4. The canonical rerun full model reaches `87.71%`.
5. Removing the graph branch lowers performance to `86.49%`.
6. Removing image information lowers performance more strongly to `81.08%`.
7. Multi-seed reruns confirm stability at `87.52% +/- 0.74%`.

This is the strongest rerun-backed Chapter 4 narrative currently supported by
the local artifacts.
