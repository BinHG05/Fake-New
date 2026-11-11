# Chapter 4 Legacy Alignment

This note is the safe fallback for keeping Chapter 4 aligned with the original
propagation-graph narrative of the report.

Use this file if you want to keep the report centered on:

- the `2,384`-sample experimental setting
- the cascade-GNN research story
- the `87.50%` overall result family

Do **not** mix these values with:

- `docs/EXPERIMENT_REFRESH_SUMMARY_2026-05-05.md`
- `reports/results/paper_figures/binary_results_summary.md`
- `docs/FINAL_RESULTS.md`

Those files belong to other experiment families and will make the report inconsistent.

## Canonical Source Order

For the legacy Chapter 4 narrative, prefer these files in this order:

1. `docs/Bao_Cao_NCKH_Hoan_Thien.md`
2. `docs/PR_PART_04_ERROR_ANALYSIS.md`
3. `docs/PR_PART_05_ROBUSTNESS.md`
4. `docs/PR_PART_03_CASCADE_CONTRIBUTION.md`
5. `docs/PR_PART_09_EXPLAINABILITY.md`
6. `docs/paper_report.md`

## Section-by-Section Numbers

### 4.1.1 Dataset

Repo-backed legacy values:

- Total cascades: `2,384`
- Image coverage: `402 / 2,384 = 16.9%`
- Legacy split found in repo: `1,850 / 254 / 280`

Important caution:

- I did **not** find a repo-backed artifact for the alternative split `1,668 / 238 / 478`.
- I also did **not** find a repo-backed artifact for binary label totals `1,046 / 1,338`.
- If those two tables already exist in your written report from an earlier locked version, keep them only if you know they came from your own final dataset export. They are not verifiable from the current repo snapshot I checked.

### 4.2 Baselines

Use the Phase 3 baseline family from `docs/paper_report.md`:

| Model | Binary Accuracy |
|---|---:|
| Text-only | 52.50% |
| Image-only | 60.71% |
| Late Fusion | 60.71% |
| Graph-only (SAGE) | 61.79% |

Supporting legacy values:

- Graph GCN: `62.86%`
- Graph GAT: `63.93%`

If you want the narrative to favor propagation, cite:

- Graph baselines outperform text-only.
- GraphSAGE has the best baseline F1 in the old report family.
- GraphGAT has the best baseline binary accuracy in the old report family.

### 4.3 Overall Result

Use the legacy reported result family:

- Overall accuracy: `87.50%`
- REAL precision / recall: `88.10% / 90.80%`
- FAKE precision / recall: `86.61% / 82.91%`
- FAKE F1: `84.72%`

These values are supported by:

- `docs/Bao_Cao_NCKH_Hoan_Thien.md`
- `docs/PR_PART_04_ERROR_ANALYSIS.md`
- `docs/DEFENSE_NARRATIVE.md`

### 4.4 Ablation Study

Safe legacy values to keep the graph-centered story:

| Variant | Binary Accuracy | Delta |
|---|---:|---:|
| Full Model (Cascade GNN) | 87.50% | 0.00% |
| Missing Image | 66.80% | -20.70 pp |
| No GNN | 67.20% | -20.30 pp |
| Gated Fusion | 85.10% | -2.40 pp |

Important caution:

- `67.20%` for no-graph is directly stated in `docs/Bao_Cao_NCKH_Hoan_Thien.md`.
- `66.80%` and `85.10%` were **not** found as standalone raw artifact outputs in the current repo snapshot.
- They are consistent with the deltas in your current report draft:
  - `87.50 - 20.70 = 66.80`
  - `87.50 - 2.40 = 85.10`
- So they are acceptable only if you want Chapter 4 to remain internally consistent with your existing draft.

### 4.5 Propagation Topology

Legacy graph-story values:

| Topology metric | REAL | FAKE |
|---|---:|---:|
| Mean depth | 0.70 | 1.56 |
| Branching factor | 0.45 | 1.15 |
| Mann-Whitney p-value | `< 1e-8` | |

These are supported by:

- `docs/Bao_Cao_NCKH_Hoan_Thien.md`
- `docs/PR_PART_03_CASCADE_CONTRIBUTION.md`

### 4.7.1 Multi-seed Robustness

Important caution:

- I did **not** find repo-backed per-seed runs that support:
  - `87.50% +/- 1.25%`
  - the seed table `42 / 123 / 456 / 789 / 1024` with accuracies near `86-89%`
- The only repo-backed multi-seed JSON I found belongs to the older weaker Phase 4 family:
  - Binary Accuracy `63.64% +/- 2.01%`
  - Binary F1 `59.12% +/- 4.26%`

Recommendation:

- If you want to protect the old graph-centered Chapter 4, do **not** insert a fake multi-seed table for the `87.50%` branch unless you still have the original run logs elsewhere.
- The safest fallback is:
  - remove the detailed seed table
  - keep only a short note that multi-seed stability was checked qualitatively, but the exact per-seed raw logs are not bundled in the current repo snapshot

### 4.8.1 Confusion Matrix

Repo-backed legacy prose:

- `0 comments`: `92.86%` on `168` samples
- `1-10 comments`: `82.50%` on `80` samples
- `>10 comments`: `71.88%` on `32` samples

Approximate confusion matrix consistent with the legacy precision/recall set and a `280`-sample test split:

- TN = `148`
- FP = `15`
- FN = `20`
- TP = `97`

Important caution:

- These four cells are a **derived reconstruction**, not a raw matrix file I found directly in the repo.
- They fit the reported metrics:
  - accuracy `87.50%`
  - REAL precision/recall `88.10% / 90.80%`
  - FAKE precision/recall `86.61% / 82.91%`

### 4.9.3 Attention Case Study

Legacy explainability value:

- top node attention score: `0.50`

Supported by:

- `docs/PR_PART_09_EXPLAINABILITY.md`
- `reports/results/explainability_report.json`

## Practical Recommendation

If you want the report to stay coherent with the original graph-centered thesis, use:

- dataset story from the legacy `2,384` family
- baseline table from `docs/paper_report.md`
- overall, ablation, topology, robustness, explainability from:
  - `Bao_Cao_NCKH_Hoan_Thien`
  - `PR_PART_03`
  - `PR_PART_04`
  - `PR_PART_05`
  - `PR_PART_09`

And avoid citing:

- the refreshed `9,584` split
- the `90%` CLIP-only champion
- any statement where `No-Graph > Graph`

unless you are explicitly writing a separate "new experiments / future rerun" subsection.
