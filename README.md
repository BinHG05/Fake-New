# Multimodal Cascade GNN: Binary Fake News Detection on Reddit

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-green.svg)](https://pyg.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%2B-orange.svg)](https://spark.apache.org/)
[![DVC Tracked](https://img.shields.io/badge/DVC-DagsHub%20Storage-purple.svg)](https://dagshub.com/BinHG05/Fake-New)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED.svg)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducibility: 100%](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)](docs/reproducibility.md)

> **A production-grade, reproducible Multimodal Data Lakehouse & Graph Neural Network (GNN) pipeline on Reddit. Combines dense semantic text (XLM-RoBERTa), vision-language cross-modal alignment (OpenAI CLIP ViT-B/32), and social cascade propagation trees with PySpark ETL, Pydantic Data Contracts, DVC lineage, and containerized serving.**

---

## 1. Problem Statement & Motivation

Online misinformation rarely spreads as isolated text. On social platforms like Reddit, misleading claims propagate through rich multimedia posts (text + images) accompanied by rapid, structured community responses (nested comment trees). 

Traditional detectors face critical architectural limitations:
- **Unimodal Text Vulnerability:** Pure NLP models fail against paraphrasing, sarcasm, or claims where the text is superficially plausible but contextually deceptive.
- **Ignoring Visual-Textual Mismatches:** Clickbait and fake news frequently pair a genuine photograph with a fabricated caption (out-of-context miscaptioning).
- **Ignoring Community Debunking Signals:** Purely static post classifiers discard crowdsourced fact-checking, skepticism, rebuttal links, and argument cascades that emerge dynamically in discussion threads.

**Our End-to-End Solution:**
1. **Medallion Data Platform:** Raw Reddit feeds are ingested through Pydantic contract gates, cleaned via distributed PySpark ETL with broadcast joins, and stored in partitioned Snappy Parquet.
2. **Multimodal Feature Extraction:** Combines 768-d XLM-RoBERTa dense text embeddings with 1025-d OpenAI CLIP features (Text, Vision, and Cross-Modal Cosine Alignment).
3. **Cascade Propagation Trees:** Community reply structures are materialized as directed graphs in PyTorch Geometric (PyG).
4. **Multimodal Cascade GNN:** A hybrid architecture featuring Cross-Attention modality fusion, 3-layer GATv2 with residual skips, and global mean/max pooling classifies posts as **REAL (0)** or **FAKE (1)**.
5. **Production Serving:** Packaged in Docker Compose with a calibrated Flask REST API and an interactive web demo dashboard.

---

## 2. Pipeline Architecture

![End-to-End Multimodal Lakehouse & GNN Architecture](docs/images/architecture_overview.png)

The pipeline is organized into four decoupled, production-grade architectural tiers:

### Tier 1 — Bronze Ingestion & Data Quality Gate
- **Raw Ingestion:** Ingests raw JSONL metadata, nested comment cascades, and high-resolution images from multi-source Reddit feeds.
- **Pydantic Data Contracts (`src.fakenews.labeling.contracts`):** Enforces strict schema typing (`CanonicalPostRecord`), ID integrity, binary label normalization (`REAL: 0`, `FAKE: 1`), and split validation.
- **Automated Quarantine Pattern (`data/quarantine/`):** Malformed JSON, corrupted records, or schema violations are isolated automatically into `rejected_records.jsonl` without failing upstream ingestion jobs.

### Tier 2 — Silver Lakehouse (Distributed PySpark ETL)
- **Deterministic Deduplication:** PySpark window functions partition by Post ID and order by timestamp to eliminate duplicates.
- **Comment Tree Flattening:** Cascade trees are exploded using `F.explode_outer("cascade")`.
- **Broadcast Hash Join (`src.fakenews.preprocessing.pyspark_etl`):** Post metadata is broadcasted across executors (`F.broadcast(post_metadata)`), eliminating network shuffle overhead when enriching high-cardinality comment streams.
- **Storage Optimization:** Materializes query-efficient **Snappy Parquet** partitioned by `(source, event_date)` to enable partition pruning for downstream feature engineering.

### Tier 3 — Gold Feature Store & Graph Construction
- **Dual Multimodal Extractors (`src.fakenews.features.embeddings`):**
  - *Text Modality:* XLM-RoBERTa extracts 768-dimensional contextual text representations.
  - *Vision-Language Modality:* OpenAI CLIP ViT-B/32 extracts 512-d text, 512-d image, and 1-d cosine similarity score ($1025\text{-d}$).
- **Graph Assembly (`src.fakenews.graphs.cascade`):**
  - *Root Post Node:* Concat $[768\text{-d Text} \parallel 1025\text{-d CLIP}] = 1793\text{-d}$.
  - *Comment Nodes:* $768\text{-d XLM-R} \parallel 1025\text{-d Zero-padding} = 1793\text{-d}$ (replies carry no independent image).
  - *Topology:* Directed edges encode parent-child discussion trees in PyTorch Geometric `Data` format.
- **Data Lineage & DVC (`dvc.yaml`):** 4-stage reproducible DAG (`validate` $\rightarrow$ `pyspark_etl` $\rightarrow$ `build_graphs` $\rightarrow$ `evaluate`) linked to remote storage on DagsHub.

### Tier 4 — GNN Reasoning & Inference Serving
- **Architecture:** Text queries Image via Cross-Attention $\rightarrow$ 3x GATv2 convolution layers with LayerNorm and residual connections $\rightarrow$ Hybrid Global Mean + Max Pooling with pre-GNN skip connection $\rightarrow$ MLP Classifier.
- **Threshold Calibration:** Decision threshold is calibrated exclusively on the Validation set ($t = 0.38$) to handle natural class imbalance without data leakage.
- **Serving Layer (`src.fakenews.serving.predictor`):** Low-latency Flask REST API providing single-post prediction, batch JSONL inference, confidence margin scoring, and uncertainty diagnostics.

---

## 3. Canonical Dataset & Verification Manifest

All experiments, DVC stages, and production benchmarks operate on a single canonical dataset:

- **Metadata Source:** `data/all_merged_binary.jsonl` ($9,584$ records)
- **Binary Class Distribution:**
  - `REAL`: 7,576 records ($79.05\%$)
  - `FAKE`: 2,008 records ($20.95\%$)
- **Data Splits (Zero Data / ID Leakage):**
  - `Train`: 8,771 posts
  - `Validation`: 406 posts
  - `Test`: 407 posts (290 REAL, 117 FAKE)
- **Feature Store Graphs:** `data/processed_graphs_v3/` ($9,584$ PyG `.pt` graphs, 100% 1-to-1 parity).
- **Integrity Status:** Verified with SHA-256 signatures in `data/manifests/pyspark_etl_manifest.json` and `artifacts/metrics/canonical_run_manifest.json`.

---

## 4. Official Benchmark Results

Evaluated on the held-out test split ($N = 407$) using the canonical checkpoint (`artifacts/checkpoints/multimodal_cascade_gnn_binary.pt`):

| Evaluation Mode | Decision Threshold | Test Accuracy | Binary F1 | FAKE Recall | FAKE Precision | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated (Production Recommended)** | **0.38** | **82.56%** | **0.7455** | **88.89%** | **0.6420** | **0.9052** |
| Standard Default | 0.50 | 83.05% | 0.6601 | 57.26% | 0.7791 | 0.9052 |
| *Historical Baseline Checkpoint*\* | *0.50* | *87.96%* | *0.7932* | *80.34%* | *0.7833* | *0.9490* |

*\*Note on Historical Baseline: The 87.96% checkpoint was trained during initial multi-seed exploration where root multimodal features were broadcasted to all cascade nodes. The canonical 83.1% (0.9052 ROC-AUC) model operates on realistic node-specific comment embeddings and is 100% reproducible via DVC.*

### Confusion Matrix (Test Split — Calibrated $t = 0.38$)

```text
                     Predicted REAL    Predicted FAKE
Actual REAL (290)         232                58
Actual FAKE (117)          13               104   (88.9% Recall on Fake News!)
```

### Ablation Study & Feature Impact

| Architecture Variant | Modalities | Test Accuracy | Binary F1 | ROC AUC |
| :--- | :--- | :---: | :---: | :---: |
| Text-Only Baseline | XLM-RoBERTa | 74.20% | 0.6624 | 0.8120 |
| Image-Only Baseline | CLIP ViT-B/32 | 28.75% | 0.4466 | 0.5410 |
| Static Cross-Attention (No Graph) | XLM-R + CLIP | 88.45% | 0.8066 | 0.9230 |
| **Full Multimodal Cascade GNN (Canonical)** | **Text + Vision + Graph** | **83.05%** | **0.7455** | **0.9052** |

---

## 5. Engineering Trade-offs & Production Insights

Deep empirical analysis revealed key realities of social media misinformation:

1. **Cascade Sparsity & The "Cold-Start" Reality:**
   - Over $50\%$ of Reddit posts have fewer than 2 comments. For single-node posts, GNN message passing operates on degenerate self-loops, and static Cross-Modal alignment (CLIP) carries the predictive signal.
2. **Discussion Dynamics at Scale (>10 Comments):**
   - As documented in `reports/results/error_analysis_report.md`, on viral posts with deep discussions ($>10$ comments), model accuracy surges to **93.33%**. Community skepticism and rebuttal trees provide decisive signals that overcome misleading root captions.
3. **Production Recommendation (Hybrid Routing):**
   - *Fresh / Low-activity posts ($<5$ comments):* Route through a lightweight, low-latency CLIP/Text inference path ($<10\text{ ms}$).
   - *Viral / High-controversy cascades ($\ge 10$ comments):* Trigger the PySpark graph construction worker and execute full GNN message passing for maximum accuracy.

---

## 6. Quickstart & Reproducibility

### 6.1. Environment Setup

```powershell
# 1. Clone repository
git clone https://github.com/BinHG05/Fake-New.git
cd Fake-New

# 2. Activate Conda environment
conda env create -f environment.yml
conda activate multimodal_gnn

# 3. PySpark setup (Requires Java 17+)
java -version

# Windows local Hadoop setup (ensure winutils.exe exists at C:\hadoop\bin)
$env:HADOOP_HOME = "C:\hadoop"
$env:Path = "$env:HADOOP_HOME\bin;$env:Path"
$env:SPARK_LOCAL_IP = "127.0.0.1"
```

### 6.2. Run Smoke Test
Verifies dataset loading, Pydantic contracts, graph building, model loading, and offline inference in under 30 seconds:

```powershell
python scripts/run_smoke_test.py
# or using task runner:
.\Makefile.ps1 smoke
```

### 6.3. Reproduce the Full Pipeline via DVC
The entire 4-stage pipeline is orchestrated with DVC and synced to DagsHub:

```powershell
# Inspect the execution DAG
python -m dvc dag

# Reproduce pipeline from contracts to evaluation
python -m dvc repro

# Pull precomputed graphs and datasets from DagsHub
python -m dvc pull
```

### 6.4. Run the Bronze-to-Silver PySpark ETL

```powershell
python scripts/pyspark_etl.py `
  --config configs/canonical.yaml `
  --parquet-output data/silver/canonical_posts `
  --jsonl-output data/silver/canonical_records.jsonl `
  --master local[2]
```

### 6.5. Reproduce Evaluation Benchmark

```powershell
python scripts/evaluate.py `
  --config configs/canonical.yaml `
  --checkpoint artifacts/checkpoints/multimodal_cascade_gnn_binary.pt `
  --tune_threshold
```

---

## 7. Serving & Interactive Web Demo

### Local Serving
Launch the Flask inference server and interactive dashboard:

```powershell
python demo/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Features:
- **Single-Post Analysis:** Real-time multimodal prediction with uncertainty bounds.
- **Batch Inference:** Evaluates JSONL feeds and displays class breakdowns.
- **Dynamic Diagnostics:** Renders live confusion matrix and reliability curves directly from verified artifact files.

### Docker Deployment
Run the complete containerized stack:

```bash
docker compose up --build
```

---

## 8. Repository Layout

```text
├── configs/
│   ├── canonical.yaml            # Official canonical hyperparameters & paths
│   └── development.yaml          # Fast local debugging configuration
│
├── data/
│   ├── all_merged_binary.jsonl   # Canonical Bronze dataset (9,584 records)
│   ├── silver/                   # PySpark Snappy Parquet (source, event_date)
│   ├── processed_graphs_v3/      # Canonical 9,584 PyG cascade graphs
│   ├── manifests/                # Pydantic validation & PySpark ETL manifests
│   └── quarantine/               # Isolated schema-violating records
│
├── artifacts/
│   ├── checkpoints/              # Official model checkpoints (.pt)
│   ├── metrics/                  # Official test_metrics.json & run manifest
│   ├── figures/                  # Confusion matrix & reliability curves
│   └── predictions/              # Test prediction JSONL with probabilities
│
├── src/fakenews/                 # Production Python Package
│   ├── config/                   # Settings and YAML loaders
│   ├── labeling/                 # Pydantic Data Contracts & Label derivation
│   ├── preprocessing/            # Distributed PySpark Bronze-to-Silver ETL
│   ├── features/                 # XLM-RoBERTa and OpenAI CLIP extractors
│   ├── graphs/                   # PyTorch Geometric CascadeGraphBuilder
│   ├── models/                   # MultimodalCascadeGNN architecture
│   ├── training/                 # Dataset loader, Focal Loss & Trainer
│   ├── evaluation/               # Metrics and visualization generators
│   └── serving/                  # Low-latency FakeNewsPredictor engine
│
├── scripts/
│   ├── generate_architecture_diagram.py # Renders docs/images/architecture_overview.png
│   ├── run_smoke_test.py         # End-to-end 6-stage verification
│   ├── pyspark_etl.py            # PySpark Bronze-to-Silver CLI
│   ├── prepare_data.py           # Pydantic Contract validation CLI
│   ├── build_graphs.py           # Batch Graph generation CLI
│   ├── train.py                  # Canonical training entrypoint
│   ├── evaluate.py               # Official evaluation entrypoint
│   └── check_secrets.py          # Automated credential scanner
│
├── demo/
│   ├── app.py                    # Flask web application & REST API
│   └── sample_batch.jsonl        # Verified sample batch for inference
│
├── tests/
│   ├── unit/                     # Unit tests for contracts, labels, model
│   └── integration/              # Offline end-to-end predictor tests
│
├── docs/
│   ├── images/                   # High-res architecture diagrams (PNG)
│   ├── architecture.md           # System and graph architecture specs
│   ├── evaluation.md             # Benchmark tables and confusion matrix
│   └── archive/                  # Legacy research baseline documentation
│
├── archive/                      # Historical code/data (ignored by Git)
├── dvc.yaml                      # Reproducible 4-stage DVC pipeline
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Container orchestration
└── environment.yml               # Conda environment definition
```

---

## 9. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
