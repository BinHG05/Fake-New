# Makefile.ps1 - PowerShell Task Runner for FakeNews Pipeline

param (
    [Parameter(Position=0)]
    [string]$Target = "help"
)

$PythonExe = "python"
if (Test-Path "D:\anacoda3\envs\multimodal_gnn\python.exe") {
    $PythonExe = "D:\anacoda3\envs\multimodal_gnn\python.exe"
}

switch ($Target.ToLower()) {
    "smoke" {
        Write-Host "Running End-to-End Smoke Test..." -ForegroundColor Cyan
        & $PythonExe scripts/run_smoke_test.py
    }
    "test" {
        Write-Host "Running Unit & Integration Tests..." -ForegroundColor Cyan
        & $PythonExe -m pytest tests/unit tests/integration
    }
    "eval" {
        Write-Host "Evaluating Canonical Checkpoint..." -ForegroundColor Cyan
        & $PythonExe scripts/evaluate.py --config configs/canonical.yaml --checkpoint artifacts/checkpoints/multimodal_cascade_gnn_binary.pt
    }
    "train" {
        Write-Host "Running Canonical Training Pipeline..." -ForegroundColor Cyan
        & $PythonExe scripts/train.py --config configs/canonical.yaml
    }
    "train-dev" {
        Write-Host "Running Fast Development Training (1 epoch, limit 32)..." -ForegroundColor Cyan
        & $PythonExe scripts/train.py --config configs/canonical.yaml --epochs 1 --limit 32
    }
    "validate-data" {
        Write-Host "Validating Canonical Dataset Contract..." -ForegroundColor Cyan
        & $PythonExe scripts/prepare_data.py --config configs/canonical.yaml
    }
    "demo" {
        Write-Host "Starting Demo Web Application on http://127.0.0.1:5000 ..." -ForegroundColor Cyan
        & $PythonExe demo/app.py
    }
    "verify" {
        Write-Host "Verifying Demo Application Endpoints..." -ForegroundColor Cyan
        & $PythonExe scripts/verify_demo.py
    }
    default {
        Write-Host "FakeNews Multimodal Cascade GNN - Task Runner" -ForegroundColor Yellow
        Write-Host "Usage: .\Makefile.ps1 <target>"
        Write-Host "Targets:"
        Write-Host "  smoke         Run end-to-end smoke test"
        Write-Host "  test          Run unit and integration test suite"
        Write-Host "  eval          Run evaluation on canonical test set"
        Write-Host "  train         Run full canonical training"
        Write-Host "  train-dev     Run quick 1-epoch development training"
        Write-Host "  validate-data Validate canonical dataset schema and contract"
        Write-Host "  demo          Launch demo web server"
        Write-Host "  verify        Run demo API verification script"
    }
}
