import sqlite3, json

conn = sqlite3.connect("training_runs.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, task_type, status, started_at, metrics FROM runs WHERE task_type LIKE '%train%' ORDER BY started_at DESC LIMIT 15"
).fetchall()
print(f"=== Training Runs ({len(rows)} total) ===\n")
for r in rows:
    m = {}
    try:
        m = json.loads(r["metrics"]) if r["metrics"] else {}
    except:
        pass
    f1 = m.get("test_f1_macro_6", "—")
    acc6 = m.get("test_acc_6", "—")
    acc_bin = m.get("test_acc_bin", "—")
    f1_bin = m.get("test_f1_bin", "—")
    print(f"  {r['id']} | {r['task_type']:20s} | {r['status']:10s} | {str(r['started_at'])[:16]}")
    print(f"    6-Class: Acc={acc6}  F1={f1}")
    print(f"    Binary:  Acc={acc_bin}  F1={f1_bin}")
    print()
conn.close()
