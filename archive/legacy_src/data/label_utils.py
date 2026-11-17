"""
Shared label utilities for binary fake-news detection.
"""

BINARY_LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1,
}

BINARY_LABEL_NAMES = ["REAL", "FAKE"]

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


def normalize_label(value: str) -> str:
    return str(value or "").strip().upper().replace("-", "_")


def derive_binary_label(record: dict) -> tuple[str, str]:
    source_dataset = str(record.get("source_dataset", "")).strip()
    current_label = normalize_label(record.get("label"))

    if source_dataset == "Fakeddit":
        original_label = normalize_label(record.get("original_label"))
        mapped = FAKEDDIT_ORIGINAL_TO_BINARY.get(original_label)
        if mapped:
            return mapped, "fakeddit_original_label"
        if current_label in BINARY_LABEL_MAP:
            return current_label, "fakeddit_binary_label"

    if source_dataset == "Reddit":
        if current_label in BINARY_LABEL_MAP:
            return current_label, "reddit_binary_label"
        mapped = REDDIT_TO_BINARY.get(current_label)
        if mapped:
            return mapped, "reddit_6class_label"

    # Safe fallback for mixed or legacy records.
    existing_binary = normalize_label(record.get("label_binary"))
    if existing_binary in BINARY_LABEL_MAP:
        return existing_binary, "existing_label_binary"

    if current_label in BINARY_LABEL_MAP:
        return current_label, "fallback_current_binary_label"

    mapped = REDDIT_TO_BINARY.get(current_label)
    if mapped:
        return mapped, "fallback_current_label"

    raise ValueError(
        f"Cannot derive binary label for id={record.get('id')} "
        f"(source_dataset={record.get('source_dataset')}, "
        f"label={record.get('label')}, original_label={record.get('original_label')}, "
        f"label_binary={record.get('label_binary')})"
    )
