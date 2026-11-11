# Multimodal Cascade GNN: Binary Fake News Detection on Reddit

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-green.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: 100%](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)](docs/reproducibility.md)

> **A reproducible, end-to-end binary fake-news detection pipeline on Reddit that integrates semantic text, visual features, and discussion cascade graphs using a Multimodal Graph Neural Network (GNN).**

---

## 1. Problem Statement & Motivation

Online misinformation rarely spreads as isolated text. On platforms like Reddit, misleading claims propagate through rich multimedia posts (text + images) accompanied by rapid, structured community responses (nested comment trees). 

Traditional detectors either:
- Rely strictly on unimodal text signals (susceptible to paraphrasing or sarcastic framing).
- Ignore discussion dynamics (missing crucial crowdsourced fact-checking, skepticism, and debunking in the comments).

**Our Solution:** We construct an end-to-end detection pipeline where:
1. Root posts fuse semantic text features (XLM-RoBERTa, 768d) with vision-language representations (CLIP ViT-B/32, 1025d).
2. Reddit reply chains are modeled as directed propagation graphs (Cascades).
3. A multi-layer Graph Attention Network (GATv2) with cross-attention fusion and hybrid pooling predicts whether a post is **REAL (0)** or **FAKE (1)**.

---

## 2. Pipeline Architecture

```text
Raw Reddit JSONL / Post
           │
           ▼
[1] Schema Validation & Binary Label Normalization (REAL: 0, FAKE: 1)
           │
           ▼
[2] Multimodal Preprocessing (Text sanitization, image resolution, comment tree parsing)
           │
           ▼
[3] Feature Extraction (XLM-RoBERTa 768d + CLIP ViT-B/32 1025d = 1793d node vector)
           │
           ▼
[4] Cascade Graph Construction (Root post + directed reply-to edges)
           │
           ▼
[5] Multimodal Cascade GNN (CrossAttention Fusion + 3x GATv2 + Hybrid Mean/Max Pooling)
           │
           ▼
[6] Evaluation & Calibration (Thresholding, reliability curve, confusion matrix)
           │
           ▼
[7] Serving & Web Demo Dashboard (Single & batch prediction with uncertainty band)
```

For complete technical specifications, see [docs/architecture.md](docs/architecture.md).

---

## 3. Canonical Dataset & Data Contract

All experiments and production benchmarks use a unified, strictly validated dataset:

- **Canonical Dataset:** `data/all_merged_binary.jsonl`
- **Total Records:** 9,584
- **Binary Distribution:**
  - `REAL`: 7,576 records (79.05%)
  - `FAKE`: 2,008 records (20.95%)
- **Data Splits (Zero ID Leakage):**
  - `Train`: 8,771 posts
  - `Validation`: 406 posts
  - `Test`: 407 posts
- **Canonical Graphs:** `data/processed_graphs_v3/` (9,584 PyG graph files, 100% 1-to-1 matching).
- **Validation Status:** 100% PASS (verified via `data/manifests/dataset_validation_report.json`).
- **Data Contract:** Detailed in [docs/data_contract.md](docs/data_contract.md).

---

## 4. Official Benchmark Results

Evaluated on the held-out test split ($N = 407$) using the official canonical checkpoint (`artifacts/checkpoints/multimodal_cascade_gnn_binary.pt`):

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Headline Test Accuracy** | **87.96%** | Primary headline accuracy |
| **Binary F1-Score** | **0.7932** | Harmonic mean for FAKE class detection |
| **Binary Precision** | **0.7833** | 78.33% of predicted fake posts are truly fake |
| **Binary Recall** | **0.8034** | 80.34% of all fake posts successfully detected |
| **ROC AUC** | **0.9490** | High separability across confidence thresholds |
| **Validation Accuracy** | **89.66%** | Validation set performance ($N = 406$) |

### Confusion Matrix (Test Split)

```text
               Predicted REAL    Predicted FAKE
Actual REAL          264               26
Actual FAKE           23               94
```

### Baseline Comparisons

| Model Architecture | Modalities Used | Test Accuracy | Binary F1 |
| :--- | :--- | :---: | :---: |
| Text-Only Baseline | Text (XLM-R) | 74.20% | 0.6624 |
| Image-Only Baseline | Vision (CLIP ViT) | 28.75% | 0.4466 |
| Early Fusion Baseline | Concat Text + Image | 75.43% | 0.6429 |
| Graph-Only Baseline | Cascade GAT (Text) | 74.20% | 0.6769 |
| CLIP-Only Benchmark | CLIP Text + Vision + Sim (1025d) | 90.00%* | 0.8772* |
| **Multimodal Cascade GNN (Canonical)** | **Text + Image + Cascade Graph** | **87.96%** | **0.7932** |

*\*Note on CLIP-only baseline: CLIP-only achieves high accuracy on static features but ignores propagation dynamics and comment debunking signals. It is preserved strictly as a feature baseline.*

Full experimental analysis is documented in [docs/evaluation.md](docs/evaluation.md).

---

## 5. Quickstart & Reproducibility

### 5.1. Setup

```powershell
# 1. Clone repository
git clone https://github.com/BinHG05/Fake-New.git
cd Fake-New

# 2. Activate Conda environment
conda env create -f environment.yml
conda activate multimodal_gnn

# 3. Setup environment variables
Copy-Item .env.example .env
```

### 5.2. Run Smoke Test
Verifies end-to-end data loading, graph building, model loading, and inference in under 30 seconds:

```powershell
python scripts/run_smoke_test.py
# or using task runner:
.\Makefile.ps1 smoke
```

### 5.3. Reproduce Evaluation
Reproduces the canonical headline metrics directly from the official checkpoint:

```powershell
python scripts/evaluate.py `
  --config configs/canonical.yaml `
  --checkpoint artifacts/checkpoints/multimodal_cascade_gnn_binary.pt
```

### 5.4. Re-train Pipeline
To train the canonical model from scratch:

```powershell
python scripts/train.py --config configs/canonical.yaml
```

To run a rapid 1-epoch debug run:
```powershell
python scripts/train.py --config configs/canonical.yaml --epochs 1 --limit 32
```

---

## 6. Lightweight Inference Demo

The project includes an inference web dashboard exposing:
- **Single Post Prediction:** Real-time multimodal analysis with uncertainty detection.
- **Batch Prediction:** Processes JSONL batches and provides summary statistics.
- **Official Metrics & Matrix:** Reads directly from verified artifact files (no hardcoded metrics).
- **Health Check:** Real-time model runtime diagnostics.

Launch the demo:
```powershell
python demo/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

To verify demo endpoints automatically:
```powershell
python scripts/verify_demo.py
```

---

## 7. Project Structure

```text
├── configs/
│   ├── canonical.yaml            # Official canonical hyperparameters
│   ├── development.yaml          # Fast local debugging configuration
│   └── demo.yaml                 # Serving and demo configuration
│
├── data/
│   ├── all_merged_binary.jsonl   # Canonical 9,584 records dataset
│   ├── processed_graphs_v3/      # Canonical 9,584 PyG cascade graphs
│   └── manifests/                # Data contract validation reports
│
├── artifacts/
│   ├── checkpoints/              # Official model weights (.pt)
│   ├── metrics/                  # Official evaluation metrics JSON
│   ├── figures/                  # Confusion matrix & reliability curves
│   └── predictions/              # Test prediction JSONL
│
├── src/fakenews/                 # Production pipeline package
│   ├── config/                   # Settings and YAML loaders
│   ├── ingestion/                # Reddit & news crawlers
│   ├── preprocessing/            # Text and image processors
│   ├── labeling/                 # Label normalizer and contract validator
│   ├── features/                 # XLM-R and CLIP extractors
│   ├── graphs/                   # Cascade graph builder
│   ├── models/                   # MultimodalCascadeGNN and baselines
│   ├── training/                 # Dataset loader and modular trainer
│   ├── evaluation/               # Metrics and plotting
│   └── serving/                  # Production predictor engine
│
├── scripts/
│   ├── run_smoke_test.py         # End-to-end smoke test
│   ├── evaluate.py               # Official evaluation entrypoint
│   ├── train.py                  # Canonical training entrypoint
│   ├── predict.py                # Prediction CLI
│   ├── prepare_data.py           # Contract validator CLI
│   ├── build_graphs.py           # Graph builder CLI
│   └── verify_demo.py            # Automated demo test suite
│
├── demo/
│   ├── app.py                    # Flask web application
│   └── sample_batch.jsonl        # Verified 10-sample batch
│
├── tests/
│   ├── unit/                     # Unit tests for labels, validator, model
│   └── integration/              # Integration test for predictor
│
├── docs/
│   ├── architecture.md           # System and graph architecture
│   ├── data_contract.md          # Data contract rules and statistics
│   ├── reproducibility.md        # Step-by-step reproduction instructions
│   ├── evaluation.md             # Benchmark tables and confusion matrix
│   ├── limitations.md            # Boundary conditions and future work
│   └── archive/                  # Legacy baseline documentation
│
├── Makefile.ps1                  # PowerShell task runner
└── environment.yml               # Conda environment definition
```

---

## 8. Limitations & Boundaries

1. **Early Propagation Cold-Start:** Newly created posts with zero comments rely on root features and skip connections; topological gains emerge as discussion develops.
2. **Platform Focus:** Cascade structures are optimized for hierarchical forum trees (Reddit); adapting to microblog reposts requires topology remapping.
3. **Missing Modality Graceful Fallback:** Posts without images are detected with zero-padded visual tokens and tagged with diagnostic warnings.

Detailed discussion in [docs/limitations.md](docs/limitations.md).

---

## 9. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
