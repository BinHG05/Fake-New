# Reproducibility Guide

> **Deterministic steps to verify and reproduce the canonical FakeNews pipeline from a clean environment.**

---

## 1. Environment Setup

### 1.1. Create and Activate Environment

```powershell
# Using Conda
conda env create -f environment.yml
conda activate multimodal_gnn

# Or install dependencies into active Python environment
pip install -e .
```

### 1.2. Configuration Setup

Copy the example environment configuration:
```powershell
Copy-Item .env.example .env
```

---

## 2. Quick Verification (Smoke Test)

Run the automated end-to-end smoke test to verify all core components (schema validation, cascade graph creation, checkpoint loading, single and batch inference):

```powershell
python scripts/run_smoke_test.py
```

Expected output:
```text
[1/6] Config loaded successfully.
[2/6] Sample dataset validated: 10 records, 0 errors.
[3/6] Cascade graph built successfully (Nodes=1, Edges=1, Dim=1793).
[4/6] Canonical model checkpoint loaded successfully (multimodal_cascade_gnn_binary.pt).
[5/6] Single inference output: {"prediction": "REAL", ...}
[6/6] Batch inference executed: 10 records processed.
============================================================
SMOKE TEST RESULT: ALL CHECKS PASSED (100% REPRODUCIBLE)
============================================================
```

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Validate Dataset Contract
Verify that the canonical dataset adheres to strict schema rules and matches the graph directory:
```powershell
python scripts/prepare_data.py --config configs/canonical.yaml
```
Output report is stored at `data/manifests/dataset_validation_report.json`.

### Step 2: Evaluate Official Canonical Checkpoint
Run evaluation on the official test split:
```powershell
python scripts/evaluate.py `
  --config configs/canonical.yaml `
  --checkpoint artifacts/checkpoints/multimodal_cascade_gnn_binary.pt
```

Outputs generated:
- `artifacts/metrics/test_metrics.json`
- `artifacts/figures/confusion_matrix.png`
- `artifacts/figures/reliability_curve.png`
- `artifacts/predictions/test_predictions.jsonl`

### Step 3: Run Full Pipeline Training
To re-train the canonical model from scratch:
```powershell
python scripts/train.py --config configs/canonical.yaml
```

To run a fast smoke training run (1 epoch, 32 samples):
```powershell
python scripts/train.py --config configs/canonical.yaml --epochs 1 --limit 32
```

### Step 4: Run Inference Demo & Verification
Launch the demo web server:
```powershell
python demo/app.py
```

In a separate terminal, execute the demo verification test suite:
```powershell
python scripts/verify_demo.py
```

---

## 4. Automation via PowerShell Task Runner (`Makefile.ps1`)

The project provides `Makefile.ps1` to run all standard targets:
```powershell
.\Makefile.ps1 smoke     # Runs smoke test
.\Makefile.ps1 test      # Runs pytest test suite
.\Makefile.ps1 eval      # Evaluates canonical checkpoint
.\Makefile.ps1 demo      # Launches demo app
.\Makefile.ps1 verify    # Runs full verification suite
```
