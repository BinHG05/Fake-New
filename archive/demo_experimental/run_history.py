"""
SQLite helper for storing training/crawl run history.
"""
import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "training_runs.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            params TEXT DEFAULT '{}',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            metrics TEXT DEFAULT '{}',
            log TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


def create_run(run_id: str, task_type: str, params: dict) -> dict:
    conn = get_conn()
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO runs (id, task_type, status, params, started_at) VALUES (?, ?, 'running', ?, ?)",
        (run_id, task_type, json.dumps(params, ensure_ascii=False), now),
    )
    conn.commit()
    conn.close()
    return {"id": run_id, "task_type": task_type, "status": "running", "started_at": now}


def append_log(run_id: str, text: str):
    conn = get_conn()
    conn.execute("UPDATE runs SET log = log || ? WHERE id = ?", (text, run_id))
    conn.commit()
    conn.close()


def finish_run(run_id: str, status: str = "completed", metrics: dict = None):
    conn = get_conn()
    now = datetime.now().isoformat()
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ?, metrics = ? WHERE id = ?",
        (status, now, json.dumps(metrics or {}, ensure_ascii=False), run_id),
    )
    conn.commit()
    conn.close()


def get_all_runs():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run(run_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_run(run_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    conn.commit()
    conn.close()


# Initialize on import
init_db()
