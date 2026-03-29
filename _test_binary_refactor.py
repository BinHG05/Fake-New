"""
Test suite for the Binary Refactor validation.
Run: python _test_binary_refactor.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def test_labeled_master_binary():
    print("\n" + "=" * 60)
    print("TEST 1: labeled_master_binary.jsonl")
    print("=" * 60)

    path = PROJECT_ROOT / "data" / "03_clean" / "Fakeddit" / "labeled_master_binary.jsonl"
    check("File exists", path.exists(), str(path))
    if not path.exists():
        return

    labels = Counter()
    splits = Counter()
    count_6class = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            labels[rec.get("label", "MISSING")] += 1
            splits[rec.get("split", "MISSING")] += 1
            if "label_6class" in rec:
                count_6class += 1

    print(f"  Total records: {total}")
    print(f"  Label distribution: {dict(labels)}")
    print(f"  Split distribution: {dict(splits)}")
    print(f"  Records with label_6class preserved: {count_6class}")

    check("Has records", total > 0, f"total={total}")
    check("All labels are REAL/FAKE", set(labels.keys()) <= {"REAL", "FAKE"}, f"found: {set(labels.keys())}")
    check("Has train split", "train" in splits)
    check("Has val split", "val" in splits)
    check("Has test split", "test" in splits)
    check("label_6class preserved", count_6class > 0, "no records have label_6class")


def test_enriched_binary():
    print("\n" + "=" * 60)
    print("TEST 2: reddit_enriched_binary.jsonl")
    print("=" * 60)

    path = PROJECT_ROOT / "data" / "reddit_enriched_binary.jsonl"
    check("File exists", path.exists(), str(path))
    if not path.exists():
        return

    labels = Counter()
    has_cascade = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            labels[rec.get("label", "MISSING")] += 1
            if rec.get("cascade"):
                has_cascade += 1

    print(f"  Total records: {total}")
    print(f"  Label distribution: {dict(labels)}")
    print(f"  Records with cascade: {has_cascade}/{total}")

    check("Has records", total > 0, f"total={total}")
    check("All labels are REAL/FAKE", set(labels.keys()) <= {"REAL", "FAKE"}, f"found: {set(labels.keys())}")
    check("Has cascade data", has_cascade > 0, "no cascades found")


def test_label_utils():
    print("\n" + "=" * 60)
    print("TEST 3: label_utils.py derive_binary_label()")
    print("=" * 60)

    from src.data.label_utils import derive_binary_label, BINARY_LABEL_MAP, BINARY_LABEL_NAMES

    check("BINARY_LABEL_MAP has REAL=0", BINARY_LABEL_MAP.get("REAL") == 0)
    check("BINARY_LABEL_MAP has FAKE=1", BINARY_LABEL_MAP.get("FAKE") == 1)
    check("BINARY_LABEL_NAMES correct", BINARY_LABEL_NAMES == ["REAL", "FAKE"], str(BINARY_LABEL_NAMES))

    # Fakeddit records
    r, _ = derive_binary_label({"source_dataset": "Fakeddit", "original_label": "True", "label": "TRUE"})
    check("Fakeddit True -> REAL", r == "REAL", r)

    r, _ = derive_binary_label({"source_dataset": "Fakeddit", "original_label": "Fake", "label": "FALSE"})
    check("Fakeddit Fake -> FAKE", r == "FAKE", r)

    # Reddit 6-class
    for lbl, expected in [("TRUE", "REAL"), ("MOSTLY_TRUE", "REAL"), ("HALF_TRUE", "REAL"),
                          ("BARELY_TRUE", "FAKE"), ("FALSE", "FAKE"), ("PANTS_ON_FIRE", "FAKE")]:
        r, _ = derive_binary_label({"source_dataset": "Reddit", "label": lbl})
        check(f"Reddit {lbl} -> {expected}", r == expected, r)

    # Reddit direct binary
    r, _ = derive_binary_label({"source_dataset": "Reddit", "label": "REAL"})
    check("Reddit REAL -> REAL", r == "REAL", r)

    r, _ = derive_binary_label({"source_dataset": "Reddit", "label": "FAKE"})
    check("Reddit FAKE -> FAKE", r == "FAKE", r)


def test_graphs_exist():
    print("\n" + "=" * 60)
    print("TEST 4: Graph Data")
    print("=" * 60)

    text_dir = PROJECT_ROOT / "data" / "processed_graphs"
    multi_dir = PROJECT_ROOT / "data" / "processed_graphs_multimodal"

    text_count = len(list(text_dir.glob("*.pt"))) if text_dir.exists() else 0
    multi_count = len(list(multi_dir.glob("*.pt"))) if multi_dir.exists() else 0

    print(f"  Text-only graphs: {text_count}")
    print(f"  Multimodal graphs: {multi_count}")

    check("Text graph dir exists", text_dir.exists())
    check("Has text graphs", text_count > 0, f"count={text_count}")
    check("Multimodal graph dir exists", multi_dir.exists())
    check("Has multimodal graphs", multi_count > 0, f"count={multi_count}")


def test_checkpoints():
    print("\n" + "=" * 60)
    print("TEST 5: Model Checkpoints")
    print("=" * 60)

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    if not ckpt_dir.exists():
        check("Checkpoint dir exists", False, str(ckpt_dir))
        return

    pts = list(ckpt_dir.glob("*.pt"))
    binary_pts = [p for p in pts if "binary" in p.name.lower()]

    print(f"  Total checkpoints: {len(pts)}")
    print(f"  Binary-named checkpoints: {len(binary_pts)}")
    for p in pts:
        size_mb = p.stat().st_size / (1024 * 1024)
        tag = " [BINARY]" if "binary" in p.name.lower() else ""
        print(f"    - {p.name} ({size_mb:.1f} MB){tag}")

    check("Has checkpoints", len(pts) > 0)


def test_consistency():
    print("\n" + "=" * 60)
    print("TEST 6: Consistency Check (master vs enriched)")
    print("=" * 60)

    master_path = PROJECT_ROOT / "data" / "03_clean" / "Fakeddit" / "labeled_master_binary.jsonl"
    enriched_path = PROJECT_ROOT / "data" / "reddit_enriched_binary.jsonl"

    if not master_path.exists() or not enriched_path.exists():
        check("Both files exist", False)
        return

    master_ids = set()
    with open(master_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                master_ids.add(str(json.loads(line.strip()).get("id", "")))

    enriched_ids = set()
    with open(enriched_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                enriched_ids.add(str(json.loads(line.strip()).get("id", "")))

    overlap = master_ids & enriched_ids
    only_master = master_ids - enriched_ids
    only_enriched = enriched_ids - master_ids

    print(f"  Master IDs: {len(master_ids)}")
    print(f"  Enriched IDs: {len(enriched_ids)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Only in master: {len(only_master)}")
    print(f"  Only in enriched: {len(only_enriched)}")

    check("Overlap exists", len(overlap) > 0, "no common IDs")
    check("No orphan enriched IDs", len(only_enriched) == 0,
          f"{len(only_enriched)} IDs in enriched but not in master")


if __name__ == "__main__":
    print("🔬 BINARY REFACTOR VALIDATION TEST SUITE")
    print("=" * 60)

    test_labeled_master_binary()
    test_enriched_binary()
    test_label_utils()
    test_graphs_exist()
    test_checkpoints()
    test_consistency()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} PASSED / {FAIL} FAILED / {PASS + FAIL} TOTAL")
    print("=" * 60)

    if FAIL > 0:
        print("⚠️  Some tests FAILED — review output above")
        sys.exit(1)
    else:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
