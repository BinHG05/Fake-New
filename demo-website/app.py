"""
Flask backend for Fake News Detection Dashboard.

Wraps existing project scripts via subprocess and provides
REST API + SSE streaming for the frontend.
"""
import os
import sys
import json
import uuid
import time
import subprocess
import threading
import re
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory

import run_history

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Use the same Python that runs Flask (from conda multimodal_gnn env)
PYTHON = sys.executable

app = Flask(__name__, static_folder=".", static_url_path="")

# ── In-memory task tracking ──
active_tasks = {}          # task_id → {"process": Popen, "log": [], "status": str}
log_subscribers = {}       # task_id → [queue, ...]


# =============================================================
# Helpers
# =============================================================

def stream_process(task_id: str, proc: subprocess.Popen, task_type: str):
    """Read stdout/stderr line-by-line from subprocess, store in log + DB."""
    log_lines = active_tasks[task_id]["log"]
    metrics = {}

    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        line = line.rstrip("\n")
        log_lines.append(line)
        run_history.append_log(task_id, line + "\n")

        # Broadcast to SSE subscribers
        if task_id in log_subscribers:
            for q in log_subscribers[task_id]:
                q.append(line)

        # Try to extract metrics from training output
        metrics = _try_parse_metrics(line, metrics)

    proc.wait()
    exit_code = proc.returncode
    status = "completed" if exit_code == 0 else "failed"
    active_tasks[task_id]["status"] = status
    run_history.finish_run(task_id, status=status, metrics=metrics)

    # Notify SSE subscribers of completion
    if task_id in log_subscribers:
        for q in log_subscribers[task_id]:
            q.append(f"[DONE] Task {status} (exit code {exit_code})")


def _try_parse_metrics(line: str, metrics: dict) -> dict:
    """Try to extract accuracy/F1 metrics from log lines."""
    # Match patterns like "6-Class Accuracy:  0.3521"
    m = re.search(r"6-Class Accuracy:\s+([\d.]+)", line)
    if m:
        metrics["test_acc_6"] = float(m.group(1))
    m = re.search(r"6-Class Macro-F1:\s+([\d.]+)", line)
    if m:
        metrics["test_f1_macro_6"] = float(m.group(1))
    m = re.search(r"Binary Accuracy:\s+([\d.]+)", line)
    if m:
        metrics["test_acc_bin"] = float(m.group(1))
    m = re.search(r"Binary F1:\s+([\d.]+)", line)
    if m:
        metrics["test_f1_bin"] = float(m.group(1))
    # Crawl stats
    m = re.search(r"New:\s*(\d+)", line)
    if m:
        metrics["new_items"] = int(m.group(1))
    # Graph build
    m = re.search(r"\[NEW\]", line)
    if m:
        metrics["graphs_created"] = metrics.get("graphs_created", 0) + 1
    m = re.search(r"\[SKIP\]", line)
    if m:
        metrics["graphs_skipped"] = metrics.get("graphs_skipped", 0) + 1
    return metrics


def launch_script(task_type: str, script_path: str, args: list = None, params: dict = None):
    """Launch a Python script as subprocess and track it."""
    task_id = str(uuid.uuid4())[:12]
    cmd = [PYTHON, "-u", script_path] + (args or [])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )

    active_tasks[task_id] = {
        "process": proc,
        "log": [],
        "status": "running",
        "task_type": task_type,
    }

    run_history.create_run(task_id, task_type, params or {})

    # Stream in background thread
    t = threading.Thread(target=stream_process, args=(task_id, proc, task_type), daemon=True)
    t.start()

    return task_id


# =============================================================
# Static files
# =============================================================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# =============================================================
# Data Pipeline APIs
# =============================================================

@app.route("/api/crawl/start", methods=["POST"])
def start_crawl():
    """Start Reddit crawler."""
    data = request.json or {}
    limit = data.get("limit", 25)
    images_only = data.get("images_only", False)
    subreddits = data.get("subreddits", None)

    script = os.path.join(PROJECT_ROOT, "src", "data", "reddit_crawler.py")
    args = ["--limit", str(limit)]
    if images_only:
        args.append("--images-only")
    if subreddits:
        args.extend(["--subreddits"] + subreddits)

    params = {"limit": limit, "images_only": images_only, "subreddits": subreddits}
    task_id = launch_script("crawl", script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/enrich/start", methods=["POST"])
def start_enrich():
    """Start crawler_enrich.py to fetch comment trees."""
    script = os.path.join(PROJECT_ROOT, "src", "data", "crawler_enrich.py")
    task_id = launch_script("enrich", script, [], {})
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/build-graphs/start", methods=["POST"])
def start_build_graphs():
    """Start build_final_graphs.py to create .pt graph files."""
    script = os.path.join(PROJECT_ROOT, "src", "utils", "build_final_graphs.py")
    task_id = launch_script("build_graphs", script, [], {})
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/visualize/start", methods=["POST"])
def start_visualize():
    """Start visualize_graphs.py to generate cascade_visualization.html."""
    script = os.path.join(PROJECT_ROOT, "src", "utils", "visualize_graphs.py")
    task_id = launch_script("visualize", script, [], {})
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/visualize/view")
def view_visualization():
    """Serve the generated cascade_visualization.html."""
    viz_path = os.path.join(PROJECT_ROOT, "data", "cascade_visualization.html")
    if os.path.exists(viz_path):
        return send_from_directory(os.path.join(PROJECT_ROOT, "data"), "cascade_visualization.html")
    return jsonify({"error": "Visualization not generated yet"}), 404


@app.route("/api/reddit-pipeline/start", methods=["POST"])
def start_reddit_pipeline():
    """Start reddit_pipeline.py — 4-step processing: Image → Text → Validate → LS."""
    data = request.json or {}
    mode = data.get("mode", "auto")  # 'auto' or 'manual'
    script = os.path.join(PROJECT_ROOT, "src", "utils", "reddit_pipeline.py")

    args = []
    if mode == "manual":
        start = data.get("start", 0)
        count = data.get("count", 50)
        args = ["--start", str(start), "--count", str(count)]
        params = {"mode": "manual", "start": start, "count": count}
    else:
        params = {"mode": "auto"}

    task_id = launch_script("reddit_pipeline", script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/batch-pipeline/start", methods=["POST"])
def start_batch_pipeline():
    """Start batch_pipeline.py for Fakeddit dataset processing."""
    data = request.json or {}
    start = data.get("start", 0)
    count = data.get("count", 200)
    script = os.path.join(PROJECT_ROOT, "src", "utils", "batch_pipeline.py")
    args = ["--start", str(start), "--count", str(count)]
    params = {"start": start, "count": count}
    task_id = launch_script("batch_pipeline", script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/merge-splits/start", methods=["POST"])
def start_merge_splits():
    """Start merge_splits.py to merge train/val/test JSONL for graph construction."""
    data = request.json or {}
    script = os.path.join(PROJECT_ROOT, "src", "utils", "merge_splits.py")
    train = data.get("train", "labels_storage/export/train_done.jsonl")
    val = data.get("val", "labels_storage/export/val_done.jsonl")
    test = data.get("test", "labels_storage/export/test_done.jsonl")
    output = data.get("output", "data/04_graph/merged_data.jsonl")
    args = ["--train", train, "--val", val, "--test", test, "--output", output]
    params = {"train": train, "val": val, "test": test, "output": output}
    task_id = launch_script("merge_splits", script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/convert-ls/start", methods=["POST"])
def start_convert_ls():
    """Convert Label Studio JSON export to JSONL."""
    data = request.json or {}
    input_file = data.get("input", "")
    merge_master = data.get("merge_master", False)
    script = os.path.join(PROJECT_ROOT, "src", "utils", "convert_ls_export_to_jsonl.py")

    if not input_file:
        return jsonify({"error": "input file path required"}), 400

    args = ["--input", input_file]
    if merge_master:
        args.append("--merge-master")

    params = {"input": input_file, "merge_master": merge_master}
    task_id = launch_script("convert_ls", script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/auto-label/start", methods=["POST"])
def start_auto_label():
    """Auto-label data using zero-shot classification."""
    data = request.json or {}
    input_file = data.get("input", "")
    mode = data.get("mode", "binary")
    method = data.get("method", "clip")
    threshold = data.get("threshold", 0.3)
    limit = data.get("limit", None)
    ls_predictions = data.get("ls_predictions", False)

    if not input_file:
        return jsonify({"error": "input file path required"}), 400

    script = os.path.join(PROJECT_ROOT, "src", "utils", "auto_labeler.py")
    args = ["--input", input_file, "--method", method, "--mode", mode,
            "--threshold", str(threshold)]

    if limit:
        args.extend(["--limit", str(limit)])
    if ls_predictions:
        args.append("--ls-predictions")

    params = {"input": input_file, "method": method, "mode": mode,
              "threshold": threshold, "limit": limit, "ls_predictions": ls_predictions}
    task_id = launch_script("auto_label", script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


@app.route("/api/auto-label/files")
def list_auto_label_files():
    """Scan project for JSONL files that can be auto-labeled."""
    import glob
    patterns = [
        os.path.join(PROJECT_ROOT, "data", "03_clean", "**", "*.jsonl"),
    ]
    files = []
    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            rel = os.path.relpath(f, PROJECT_ROOT).replace("\\", "/")
            size_kb = round(os.path.getsize(f) / 1024, 1)
            # Count lines
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    line_count = sum(1 for _ in fh)
            except Exception:
                line_count = 0
            files.append({"path": rel, "size_kb": size_kb, "lines": line_count})
    files.sort(key=lambda x: x["path"])
    return jsonify(files)


@app.route("/api/ls-exports")
def list_ls_exports():
    """Scan project for Label Studio export JSON files."""
    import glob
    patterns = [
        os.path.join(PROJECT_ROOT, "data", "03_clean", "**", "export_*.json"),
        os.path.join(PROJECT_ROOT, "data", "03_clean", "**", "*_for_ls.json"),
        os.path.join(PROJECT_ROOT, "labels_storage", "export", "*.json"),
    ]
    files = []
    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            rel = os.path.relpath(f, PROJECT_ROOT).replace("\\", "/")
            size_kb = round(os.path.getsize(f) / 1024, 1)
            files.append({"path": rel, "size_kb": size_kb})
    # Sort by path
    files.sort(key=lambda x: x["path"])
    return jsonify(files)

@app.route("/api/train/start", methods=["POST"])
def start_training():
    """Start a training run. Accepts model_type + hyperparams."""
    data = request.json or {}
    model_type = data.get("model_type", "text")
    epochs = data.get("epochs", 20)
    lr = data.get("lr", 2e-4)
    batch_size = data.get("batch_size", 16)
    patience = data.get("patience", 7)

    # Route to correct script
    if model_type in ("text", "image", "fusion"):
        script = os.path.join(PROJECT_ROOT, "src", "training", "train_baseline.py")
        args = [
            "--model_type", model_type,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch_size", str(batch_size),
            "--patience", str(patience),
        ]
        task_type = f"train_{model_type}"

    elif model_type == "gnn":
        script = os.path.join(PROJECT_ROOT, "src", "training", "train_gnn.py")
        gnn_type = data.get("gnn_type", "gcn")
        args = [
            "--epochs", str(epochs),
            "--gnn_type", gnn_type,
        ]
        task_type = "train_gnn"

    elif model_type == "multimodal":
        script = os.path.join(PROJECT_ROOT, "src", "training", "train_multimodal_gnn.py")
        fusion_type = data.get("fusion_type", "cross_attention")
        fusion_dim = data.get("fusion_dim", 256)
        args = [
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--fusion_type", fusion_type,
            "--fusion_dim", str(fusion_dim),
            "--patience", str(patience),
        ]
        task_type = "train_multimodal"

    else:
        return jsonify({"error": f"Unknown model_type: {model_type}"}), 400

    params = {k: v for k, v in data.items()}
    task_id = launch_script(task_type, script, args, params)
    return jsonify({"task_id": task_id, "status": "started"})


# =============================================================
# Task Status & Streaming
# =============================================================

@app.route("/api/task/<task_id>/status")
def task_status(task_id):
    """Return current status + last N log lines for a task."""
    if task_id in active_tasks:
        t = active_tasks[task_id]
        return jsonify({
            "task_id": task_id,
            "status": t["status"],
            "log_lines": t["log"][-100:],
            "total_lines": len(t["log"]),
        })
    # Check DB
    run = run_history.get_run(task_id)
    if run:
        return jsonify({
            "task_id": task_id,
            "status": run["status"],
            "log_lines": run["log"].split("\n")[-100:] if run["log"] else [],
            "total_lines": len(run["log"].split("\n")) if run["log"] else 0,
        })
    return jsonify({"error": "Not found"}), 404


@app.route("/api/task/<task_id>/stop", methods=["POST"])
def stop_task(task_id):
    """Stop a running task."""
    if task_id in active_tasks:
        proc = active_tasks[task_id]["process"]
        if proc.poll() is None:
            proc.terminate()
            active_tasks[task_id]["status"] = "stopped"
            run_history.finish_run(task_id, status="stopped")
            return jsonify({"status": "stopped"})
    return jsonify({"error": "Task not running"}), 404


@app.route("/api/stream/<task_id>")
def stream_log(task_id):
    """SSE endpoint for real-time log streaming."""
    def generate():
        q = []
        if task_id not in log_subscribers:
            log_subscribers[task_id] = []
        log_subscribers[task_id].append(q)

        # First send existing lines
        if task_id in active_tasks:
            for line in active_tasks[task_id]["log"]:
                yield f"data: {json.dumps({'line': line})}\n\n"

        # Then stream new lines
        try:
            while True:
                if q:
                    line = q.pop(0)
                    if line.startswith("[DONE]"):
                        yield f"data: {json.dumps({'line': line, 'done': True})}\n\n"
                        break
                    yield f"data: {json.dumps({'line': line})}\n\n"
                else:
                    time.sleep(0.3)
                    # Check if task finished while we were waiting
                    if task_id in active_tasks and active_tasks[task_id]["status"] != "running":
                        yield f"data: {json.dumps({'line': '[DONE]', 'done': True})}\n\n"
                        break
        finally:
            if task_id in log_subscribers and q in log_subscribers[task_id]:
                log_subscribers[task_id].remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# =============================================================
# Run History
# =============================================================

@app.route("/api/runs")
def list_runs():
    runs = run_history.get_all_runs()
    return jsonify(runs)


@app.route("/api/runs/<run_id>")
def get_run_detail(run_id):
    run = run_history.get_run(run_id)
    if run:
        return jsonify(run)
    return jsonify({"error": "Not found"}), 404


@app.route("/api/metrics/best")
def get_best_metrics():
    """Return the best recorded metrics for each model type to populate the charts."""
    runs = run_history.get_all_runs()
    
    best_stats = {
        "text": {"f1": 0, "acc": 0, "bin_acc": 0, "bin_f1": 0},
        "image": {"f1": 0, "acc": 0, "bin_acc": 0, "bin_f1": 0},
        "graph": {"f1": 0, "acc": 0, "bin_acc": 0, "bin_f1": 0},
        "fusion": {"f1": 0, "acc": 0, "bin_acc": 0, "bin_f1": 0},
        "multimodal": {"f1": 0, "acc": 0, "bin_acc": 0, "bin_f1": 0}
    }
    
    # Defaults in case nothing was ever trained
    default_f1 = {"text": 0.129, "graph": 0.190, "image": 0.102, "fusion": 0.102, "multimodal": 0.172}
    default_acc = {"text": 26.4, "graph": 23.6, "image": 43.9, "fusion": 43.9, "multimodal": 27.2}
    default_bin_acc = {"text": 52.5, "graph": 61.8, "image": 60.7, "fusion": 60.7, "multimodal": 60.3}
    default_bin_f1 = {"text": 0.561, "graph": 0.590, "image": 0.0, "fusion": 0.0, "multimodal": 0.551}
    
    for run in runs:
        if run["status"] != "completed":
            continue
            
        task_type = run.get("task_type", "")
        # Parse metrics safely
        try:
            m = json.loads(run.get("metrics", "{}")) if isinstance(run.get("metrics"), str) else run.get("metrics", {})
        except Exception:
            m = {}
            
        if not m:
            continue
            
        f1 = m.get("test_f1_macro_6", 0)
        acc = m.get("test_acc_6", 0) * 100  # multiply by 100 for percentage
        bin_acc = m.get("test_acc_bin", 0) * 100
        bin_f1 = m.get("test_f1_bin", 0)
        
        # Map task_type to chart labels
        cat = None
        if task_type == "train_text":
            cat = "text"
        elif task_type == "train_image":
            cat = "image"
        elif task_type == "train_gnn":
            cat = "graph"
        elif task_type == "train_fusion":
            cat = "fusion"
        elif task_type == "train_multimodal":
            cat = "multimodal"
            
        if cat:
            best_stats[cat]["f1"] = max(best_stats[cat]["f1"], f1)
            best_stats[cat]["acc"] = max(best_stats[cat]["acc"], acc)
            best_stats[cat]["bin_acc"] = max(best_stats[cat]["bin_acc"], bin_acc)
            best_stats[cat]["bin_f1"] = max(best_stats[cat]["bin_f1"], bin_f1)
            
    # Apply defaults if 0
    for k in best_stats:
        if best_stats[k]["f1"] == 0:
            best_stats[k]["f1"] = default_f1[k]
        if best_stats[k]["acc"] == 0:
            best_stats[k]["acc"] = default_acc[k]
        if best_stats[k]["bin_acc"] == 0:
            best_stats[k]["bin_acc"] = default_bin_acc[k]
        if best_stats[k]["bin_f1"] == 0:
            best_stats[k]["bin_f1"] = default_bin_f1[k]
            
    return jsonify({
        "labels": ["Text-Only", "Graph SAGE", "Image-Only", "Simple Fusion", "Multimodal GNN"],
        "f1Values": [best_stats["text"]["f1"], best_stats["graph"]["f1"], best_stats["image"]["f1"], best_stats["fusion"]["f1"], best_stats["multimodal"]["f1"]],
        "accValues": [best_stats["text"]["acc"], best_stats["graph"]["acc"], best_stats["image"]["acc"], best_stats["fusion"]["acc"], best_stats["multimodal"]["acc"]],
        "binAccValues": [best_stats["text"]["bin_acc"], best_stats["graph"]["bin_acc"], best_stats["image"]["bin_acc"], best_stats["fusion"]["bin_acc"], best_stats["multimodal"]["bin_acc"]],
        "binF1Values": [best_stats["text"]["bin_f1"], best_stats["graph"]["bin_f1"], best_stats["image"]["bin_f1"], best_stats["fusion"]["bin_f1"], best_stats["multimodal"]["bin_f1"]]
    })


@app.route("/api/runs/<run_id>", methods=["DELETE"])
def delete_run(run_id):
    run_history.delete_run(run_id)
    return jsonify({"status": "deleted"})


# =============================================================
# Data Stats
# =============================================================

@app.route("/api/data/stats")
def data_stats():
    """Return statistics about current datasets."""
    stats = {}

    # Raw Reddit data
    raw_file = os.path.join(PROJECT_ROOT, "data", "01_raw", "reddit", "reddit_realtime_data.jsonl")
    if os.path.exists(raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            stats["raw_reddit_count"] = sum(1 for _ in f)
    else:
        stats["raw_reddit_count"] = 0

    # Enriched data
    enriched_file = os.path.join(PROJECT_ROOT, "data", "reddit_enriched_data.jsonl")
    if os.path.exists(enriched_file):
        with open(enriched_file, "r", encoding="utf-8") as f:
            stats["enriched_count"] = sum(1 for _ in f)
    else:
        stats["enriched_count"] = 0

    # Processed graphs
    graph_dir = os.path.join(PROJECT_ROOT, "data", "processed_graphs")
    if os.path.isdir(graph_dir):
        stats["graph_count"] = len([f for f in os.listdir(graph_dir) if f.endswith(".pt")])
    else:
        stats["graph_count"] = 0

    # Checkpoints
    ckpt_dir = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    if os.path.isdir(ckpt_dir):
        stats["checkpoints"] = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    else:
        stats["checkpoints"] = []

    # Visualization exists?
    viz_path = os.path.join(PROJECT_ROOT, "data", "cascade_visualization.html")
    stats["visualization_exists"] = os.path.exists(viz_path)

    return jsonify(stats)


# =============================================================
# Entry point
# =============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Fake News Detection Dashboard")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Python:  {PYTHON}")
    print("  Open http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
