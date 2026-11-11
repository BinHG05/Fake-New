"""
General helper functions for data manipulation and environment configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    """Writes an iterable of records to a JSONL file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    return count

