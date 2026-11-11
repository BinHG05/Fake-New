"""
Canonical Flask Web Application for FakeNews Multimodal GNN Demo.
Strictly serves inference, official metrics, confusion matrix, and pipeline health.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, render_template_string
from werkzeug.utils import secure_filename

from src.fakenews.config.settings import load_config
from src.fakenews.serving.predictor import FakeNewsPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("DemoApp")

# Load configuration
config = load_config("configs/demo.yaml")
srv_cfg = config.get("serving", {})
m_cfg = config.get("model", {})

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# Initialize Predictor
predictor: FakeNewsPredictor | None = None
try:
    predictor = FakeNewsPredictor(config_path="configs/demo.yaml")
    logger.info("FakeNewsPredictor successfully initialized in demo app.")
except Exception as e:
    logger.error("Failed to initialize FakeNewsPredictor: %s", e)

# HTML UI Template
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fake News Detection Pipeline | Multimodal Cascade GNN</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --accent: #38bdf8;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --real: #22c55e;
      --fake: #ef4444;
      --uncertain: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.5; }
    .container { max-width: 1200px; margin: 0 auto; }
    header { margin-bottom: 2rem; border-bottom: 1px solid var(--card-border); padding-bottom: 1rem; }
    h1 { font-size: 1.8rem; font-weight: 700; color: var(--accent); }
    .subtitle { color: var(--text-muted); font-size: 0.95rem; margin-top: 0.25rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 1.5rem; }
    .card h2 { font-size: 1.2rem; margin-bottom: 1rem; color: var(--text); border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 0.75rem; font-size: 0.95rem; }
    .stat-label { color: var(--text-muted); }
    .stat-value { font-weight: 600; }
    .badge { padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700; font-size: 0.85rem; display: inline-block; }
    .badge-real { background: rgba(34, 197, 94, 0.2); color: var(--real); border: 1px solid var(--real); }
    .badge-fake { background: rgba(239, 68, 68, 0.2); color: var(--fake); border: 1px solid var(--fake); }
    .badge-uncertain { background: rgba(245, 158, 11, 0.2); color: var(--uncertain); border: 1px solid var(--uncertain); }
    input[type="text"], textarea { width: 100%; background: #0b1120; border: 1px solid var(--card-border); color: var(--text); padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.9rem; }
    textarea { height: 90px; resize: vertical; }
    button { background: var(--accent); color: #0b1120; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
    button:hover { opacity: 0.9; }
    .figure-container { text-align: center; margin-top: 1rem; }
    .figure-container img { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--card-border); }
    .warning-box { background: rgba(245, 158, 11, 0.1); border-left: 4px solid var(--uncertain); padding: 0.5rem 1rem; margin-top: 0.5rem; font-size: 0.85rem; color: #fbbf24; }
    pre { background: #0b1120; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; color: #a5f3fc; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Multimodal Cascade GNN Fake News Detector</h1>
      <p class="subtitle">End-to-End Binary Classification on Reddit with Text, Image & Propagation Graph</p>
    </header>

    <div class="grid">
      <!-- Left Column: Inference UI -->
      <div class="card">
        <h2>Single Post Inference</h2>
        <form id="singleForm">
          <label style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Post Text / Claim:</label>
          <textarea id="postText" placeholder="Enter post text to analyze..."></textarea>
          
          <label style="font-size: 0.85rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Optional Image Path or URL:</label>
          <input type="text" id="postImage" placeholder="e.g. data/sample.jpg (optional)">
          
          <button type="submit">Analyze Post</button>
        </form>

        <div id="singleResult" style="margin-top: 1.5rem; display: none;">
          <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">Result:</h3>
          <div class="stat-row">
            <span class="stat-label">Prediction:</span>
            <span class="stat-value" id="resBadge"></span>
          </div>
          <div class="stat-row">
            <span class="stat-label">Fake Probability:</span>
            <span class="stat-value" id="resProbFake"></span>
          </div>
          <div class="stat-row">
            <span class="stat-label">Real Probability:</span>
            <span class="stat-value" id="resProbReal"></span>
          </div>
          <div class="stat-row">
            <span class="stat-label">Decision Margin:</span>
            <span class="stat-value" id="resMargin"></span>
          </div>
          <div id="resWarnings"></div>
        </div>

        <h2 style="margin-top: 2rem;">Batch Inference</h2>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
          Run batch prediction on curated sample batch (demo/sample_batch.jsonl)
        </p>
        <button id="runBatchBtn" style="background: #3b82f6; color: white;">Run Sample Batch</button>
        <div id="batchResult" style="margin-top: 1rem; display: none;">
          <pre id="batchOutput"></pre>
        </div>
      </div>

      <!-- Right Column: Official Metrics & Pipeline Health -->
      <div class="card">
        <h2>Official Canonical Metrics</h2>
        <div id="metricsBlock">
          <p style="color: var(--text-muted); font-size: 0.9rem;">Loading official test metrics...</p>
        </div>

        <h2 style="margin-top: 1.5rem;">Confusion Matrix</h2>
        <div class="figure-container">
          <img src="/api/figures/confusion_matrix" alt="Confusion Matrix" onerror="this.style.display='none'; document.getElementById('cmFallback').style.display='block';">
          <p id="cmFallback" style="display:none; color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">Confusion matrix figure pending generation via scripts/evaluate.py</p>
        </div>

        <h2 style="margin-top: 1.5rem;">Pipeline Health</h2>
        <div id="healthBlock">
          <p style="color: var(--text-muted); font-size: 0.9rem;">Checking pipeline status...</p>
        </div>
      </div>
    </div>
  </div>

  <script>
    // Load Health
    fetch('/api/health')
      .then(r => r.json())
      .then(d => {
        document.getElementById('healthBlock').innerHTML = `
          <div class="stat-row"><span class="stat-label">Status:</span><span class="badge ${d.status === 'HEALTHY' ? 'badge-real' : 'badge-fake'}">${d.status}</span></div>
          <div class="stat-row"><span class="stat-label">Checkpoint:</span><span class="stat-value">${d.checkpoint}</span></div>
          <div class="stat-row"><span class="stat-label">Device:</span><span class="stat-value">${d.device}</span></div>
          <div class="stat-row"><span class="stat-label">Threshold:</span><span class="stat-value">${d.threshold}</span></div>
          <div class="stat-row"><span class="stat-label">Uncertain Band:</span><span class="stat-value">±${d.uncertain_threshold}</span></div>
        `;
      });

    // Load Official Metrics
    fetch('/api/metrics')
      .then(r => r.json())
      .then(d => {
        if (d.error) {
          document.getElementById('metricsBlock').innerHTML = `<p style="color: var(--uncertain);">${d.error}</p>`;
        } else {
          document.getElementById('metricsBlock').innerHTML = `
            <div class="stat-row"><span class="stat-label">Test Accuracy:</span><span class="stat-value" style="color: var(--accent); font-size: 1.1rem;">${(d.accuracy * 100).toFixed(2)}%</span></div>
            <div class="stat-row"><span class="stat-label">Binary F1-Score:</span><span class="stat-value">${d.f1.toFixed(4)}</span></div>
            <div class="stat-row"><span class="stat-label">Binary Precision:</span><span class="stat-value">${d.precision.toFixed(4)}</span></div>
            <div class="stat-row"><span class="stat-label">Binary Recall:</span><span class="stat-value">${d.recall.toFixed(4)}</span></div>
            <div class="stat-row"><span class="stat-label">ROC AUC:</span><span class="stat-value">${d.roc_auc ? d.roc_auc.toFixed(4) : 'N/A'}</span></div>
            <div class="stat-row"><span class="stat-label">Evaluated Checkpoint:</span><span class="stat-value" style="font-size: 0.8rem;">${d.checkpoint}</span></div>
          `;
        }
      });

    // Single Prediction Form
    document.getElementById('singleForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = document.getElementById('postText').value.trim();
      const image = document.getElementById('postImage').value.trim();
      if (!text) return alert('Please enter post text.');

      const resp = await fetch('/api/predict/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clean_text: text, image: image || null })
      });
      const data = await resp.json();

      document.getElementById('singleResult').style.display = 'block';
      const badge = document.getElementById('resBadge');
      badge.textContent = data.prediction;
      badge.className = 'badge ' + (data.prediction === 'REAL' ? 'badge-real' : (data.prediction === 'FAKE' ? 'badge-fake' : 'badge-uncertain'));
      document.getElementById('resProbFake').textContent = (data.probability_fake * 100).toFixed(2) + '%';
      document.getElementById('resProbReal').textContent = (data.probability_real * 100).toFixed(2) + '%';
      document.getElementById('resMargin').textContent = data.margin_to_threshold.toFixed(4);

      const warnDiv = document.getElementById('resWarnings');
      warnDiv.innerHTML = '';
      if (data.warnings && data.warnings.length > 0) {
        warnDiv.innerHTML = `<div class="warning-box">⚠️ Warnings: ${data.warnings.join(', ')}</div>`;
      }
    });

    // Run Batch Button
    document.getElementById('runBatchBtn').addEventListener('click', async () => {
      const resp = await fetch('/api/predict/batch', { method: 'POST' });
      const data = await resp.json();
      document.getElementById('batchResult').style.display = 'block';
      document.getElementById('batchOutput').textContent = JSON.stringify(data, null, 2);
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/health", methods=["GET"])
def health():
    if predictor is None:
        return jsonify({"status": "ERROR", "message": "Predictor uninitialized"}), 500
    return jsonify(predictor.health())


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_path = PROJECT_ROOT / srv_cfg.get(
        "metrics_path",
        "artifacts/metrics/test_metrics.json",
    )
    if not metrics_path.exists():
        return jsonify({"error": "test_metrics.json not found. Run scripts/evaluate.py first."}), 404

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.route("/api/figures/confusion_matrix", methods=["GET"])
def get_confusion_matrix():
    cm_path = PROJECT_ROOT / srv_cfg.get(
        "confusion_matrix_path",
        "artifacts/figures/confusion_matrix.png",
    )
    if not cm_path.exists():
        return jsonify({"error": "confusion_matrix.png not found"}), 404
    return send_file(str(cm_path), mimetype="image/png")


@app.route("/api/predict/record", methods=["POST"])
def predict_record():
    if predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 500

    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    res = predictor.predict_record(data)
    return jsonify(res)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    if predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 500

    # Default to sample batch if no file uploaded
    sample_path = PROJECT_ROOT / srv_cfg.get(
        "sample_batch_path",
        "demo/sample_batch.jsonl",
    )
    if not sample_path.exists():
        return jsonify({"error": f"Sample batch not found: {sample_path}"}), 404

    res = predictor.predict_batch(sample_path)
    return jsonify(res)


if __name__ == "__main__":
    host = str(srv_cfg.get("host", "127.0.0.1"))
    port = int(srv_cfg.get("port", 5000))
    debug = bool(srv_cfg.get("debug", False))  # False for production portfolio mode

    logger.info("Starting Demo Web Server on http://%s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)

