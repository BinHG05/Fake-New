<div align="center">
  <h1>📰 Multimodal Fake News Detection & Data Pipeline</h1>
  <p>
    <i>An end-to-end Data Engineering and Deep Learning pipeline for detecting misinformation using Unstructured Data (Text, Image) and Graph Propagation Networks.</i>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Data_Version_Control-(DVC)-orange.svg" alt="DVC">
    <img src="https://img.shields.io/badge/PyTorch-Geometric-red.svg" alt="PyG">
    <img src="https://img.shields.io/badge/Docker-Enabled-blue.svg" alt="Docker">
  </p>
</div>

---

## 🎯 About The Project

This project builds a highly scalable data pipeline and a Graph Neural Network (GNN) designed to detect fake news on social media (specifically Reddit). Rather than relying solely on text or images, it models the **information cascade**—how users interact, comment, and share data over time.

Built with a strong focus on **DataOps** and **Data Quality**, this repository heavily emphasizes the data engineering lifecycle: from reliable web crawling to multimodel feature extraction, annotation via Label-Studio, and automated packaging.

## ✨ Key Features & Pipeline Architecture

### 1. 🕷️ Fault-Tolerant Data Ingestion (Crawling)
- Automated Reddit crawler interacting robustly with Reddit API.
- Implements `urllib3` retry strategies and **Exponential Backoff** to gracefully handle HTTP 429 Rate Limits.
- Recursively parses deep comment trees to generate structural Graph/Cascade nodes.

### 2. 🧬 Multimodal ETL Pipeline
- **Text Processing**: Noise removal, tokenization, and vectorization using `XLM-RoBERTa` (768d embeddings).
- **Image Processing**: Batch downloading, structural validation (MIME-types), and Resizing/Padding (224x224) via Pillow/OpenCV for downstream CNNs (`ResNet50`).
- **Optimization**: Scripted memory efficiency to handle large unstructured blobs natively.

### 3. 🏗️ Data Architecture & Modeling
- Centralized Data Contract through a meticulously designed [schema.md](schema.md) standardizing `JSONLines` structures.
- Implements validation scripts to enforce constraints across Node, Edge, Text, and Media entities before model training.

### 4. 🗄️ DataOps, Versioning & Collaboration
- Containerized annotation process via **Docker Compose** feeding into `Label-Studio`.
- Orchestrates high-volume datasets using **Data Version Control (DVC)**, enabling reproducible AI experiments.
- Includes automated packaging scripts (`package_data.py`) to systematically export dataset snapshots for dashboard/frontend interfaces.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (Recommended)
- [DVC](https://dvc.org/)
- Docker & Docker-Compose (For Label Studio)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BinHG05/[YOUR-REPO-NAME].git
   cd [YOUR-REPO-NAME]
   ```

2. **Set up the virtual environment:**
   ```bash
   conda env create -f environment.yml
   conda activate multimodal_gnn
   ```

3. **Pull tracking datasets via DVC:**
   ```bash
   dvc pull
   ```

4. **Start Label Studio (Optional - for data annotation):**
   ```bash
   docker-compose up -d
   ```

---

## 📂 Project Structure

```text
├── data/                       # Local raw & processed data (Tracked by DVC)
├── demo-website/               # Real-time visualization dashboard (Flask/HTML)
├── docs/                       # Technical specs, schemas, and research reports
├── models/                     # PyTorch checkpoints and model configurations
├── src/
│   ├── data/                   # Crawlers, ETL pipelines, Image/Text Processors
│   ├── experiments/            # Visualization and evaluation runners
│   ├── models/                 # Neural Network architectures (GNN, CNN, BERT)
│   └── training/               # Model training loops
├── docker-compose.yml          # Label-Studio annotation environment configuration
├── environment.yml             # Conda environment build file
├── package_data.py             # Data distribution and zipping script
└── schema.md                   # JSONLines Data Contract Definition
```

---

## 🔬 Model Performance (GNN)

The underlying Machine Learning model unifies Cross-Attention Mechanisms and GraphSAGE to operate on the extracted cascade data:

* **Binary Authenticity:** Achieved **66.4%** accuracy identifying Fake vs Non-Fake cascades.
* **Granular Levels:** Detects 6 fine-grained levels of truthfulness (True, Mostly True, Half True, Barely True, False, Pants on Fire).

*Refer to [docs/paper_report.md](docs/paper_report.md) for extensive mathematical ablation and metric studies.*

---

## 👨‍💻 Author

**Nguyen Subin**  
Data Engineer / Pipeline Architect  
* [LinkedIn](https://linkedin.com/in/nguyensubin)
* [GitHub](https://github.com/BinHG05)