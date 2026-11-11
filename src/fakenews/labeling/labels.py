"""
Label definition and normalization utilities for binary fake news detection.
"""

from __future__ import annotations

from typing import Any, Tuple

BINARY_LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1,
}

BINARY_LABEL_NAMES = ["REAL", "FAKE"]

# Legacy label mappings to canonical binary
REDDIT_TO_BINARY = {
    "TRUE": "REAL",
    "MOSTLY_TRUE": "REAL",
    "HALF_TRUE": "REAL",
    "BARELY_TRUE": "FAKE",
    "FALSE": "FAKE",
    "PANTS_ON_FIRE": "FAKE",
    "PANTS_FIRE": "FAKE",
}

FAKEDDIT_ORIGINAL_TO_BINARY = {
    "TRUE": "REAL",
    "FAKE": "FAKE",
}


def normalize_label_str(value: Any) -> str:
    """Normalize label string into standardized uppercase format."""
    return str(value or "").strip().upper().replace("-", "_")


def derive_binary_label(record: dict[str, Any]) -> Tuple[str, int]:
    """
    Derives canonical binary label ('REAL' or 'FAKE') and integer index (0 or 1).
    Raises ValueError if label cannot be mapped or is Unlabeled.
    """
    # 1. Direct label_binary field
    raw_bin = normalize_label_str(record.get("label_binary"))
    if raw_bin in BINARY_LABEL_MAP:
        return raw_bin, BINARY_LABEL_MAP[raw_bin]

    # 2. Check label field
    lbl = normalize_label_str(record.get("label"))
    if lbl in BINARY_LABEL_MAP:
        return lbl, BINARY_LABEL_MAP[lbl]

    # 3. Check original_label (Fakeddit)
    orig_lbl = normalize_label_str(record.get("original_label"))
    if orig_lbl in FAKEDDIT_ORIGINAL_TO_BINARY:
        mapped = FAKEDDIT_ORIGINAL_TO_BINARY[orig_lbl]
        return mapped, BINARY_LABEL_MAP[mapped]

    # 4. Check 6-class Reddit label
    if lbl in REDDIT_TO_BINARY:
        mapped = REDDIT_TO_BINARY[lbl]
        return mapped, BINARY_LABEL_MAP[mapped]

    lbl_6 = normalize_label_str(record.get("label_6class"))
    if lbl_6 in REDDIT_TO_BINARY:
        mapped = REDDIT_TO_BINARY[lbl_6]
        return mapped, BINARY_LABEL_MAP[mapped]

    raise ValueError(
        f"Cannot derive binary label for record id={record.get('id')}. "
        f"label={record.get('label')}, label_binary={record.get('label_binary')}, "
        f"original_label={record.get('original_label')}"
    )

