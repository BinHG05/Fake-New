"""
Merge all datasets (Reddit + Voz + News) into a single master file
for graph building and training.

- Reddit: Already enriched with cascade & labels
- Voz: Labeled file (no cascade) + Raw file (has cascade) → merge cascade back
- News: All labeled REAL (báo chính thống)
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_label(label):
    """Normalize any label format to REAL/FAKE."""
    if label is None:
        return None
    label_str = str(label).strip().upper()
    if label_str in ("TRUE", "REAL", "1", "TRUE_NEWS"):
        return "REAL"
    elif label_str in ("FALSE", "FAKE", "0", "FAKE_NEWS"):
        return "FAKE"
    return None


def main():
    parser = argparse.ArgumentParser(description="Merge Reddit + Voz + News into master file")
    parser.add_argument("--reddit", default="data/reddit_enriched_binary.jsonl",
                        help="Reddit enriched binary JSONL")
    parser.add_argument("--voz-labeled", default="data/03_clean/voz/voz_run_001/voz/train_auto_labeled.jsonl",
                        help="Voz auto-labeled JSONL (from auto_labeler)")
    parser.add_argument("--voz-raw", default="data/01_raw/voz/voz_realtime_data.jsonl",
                        help="Voz raw JSONL (contains cascade)")
    parser.add_argument("--news", default="data/01_raw/news/news_realtime_data.jsonl",
                        help="News raw JSONL (báo chính thống, all REAL)")
    parser.add_argument("--output", default="data/all_merged_binary.jsonl",
                        help="Output merged JSONL")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATASET MERGER - Reddit + Voz + News")
    print("=" * 60)

    all_records = []
    label_counts = Counter()
    source_counts = Counter()
    seen_ids = set()

    # ─────────────────────────────────────────────
    # 1. REDDIT (already enriched with cascade)
    # ─────────────────────────────────────────────
    reddit_path = Path(args.reddit)
    if reddit_path.exists():
        reddit = read_jsonl(reddit_path)
        for r in reddit:
            lb = r.get("label_binary") or normalize_label(r.get("label"))
            if not lb:
                continue
            r["label_binary"] = lb
            r["label"] = lb
            r.setdefault("cascade", [])
            r.setdefault("source_dataset", "reddit")
            r.setdefault("raw_text", r.get("clean_text", ""))
            rid = r.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_records.append(r)
                label_counts[lb] += 1
                source_counts["Reddit"] += 1
        print(f"  [Reddit]  {source_counts['Reddit']} records loaded")
    else:
        print(f"  [Reddit]  File not found: {reddit_path}")

    # ─────────────────────────────────────────────
    # 2. VOZ (labeled + cascade from raw)
    # ─────────────────────────────────────────────
    voz_labeled_path = Path(args.voz_labeled)
    voz_raw_path = Path(args.voz_raw)

    if voz_labeled_path.exists():
        voz_labeled = read_jsonl(voz_labeled_path)

        # Build cascade lookup from raw file
        cascade_lookup = {}
        media_lookup = {}
        if voz_raw_path.exists():
            voz_raw = read_jsonl(voz_raw_path)
            for r in voz_raw:
                rid = r.get("id")
                if rid:
                    if "cascade" in r:
                        cascade_lookup[rid] = r["cascade"]
                    if "media_url" in r:
                        media_lookup[rid] = r["media_url"]
            print(f"  [Voz Raw] {len(voz_raw)} records scanned for cascade lookup")

        voz_added = 0
        voz_skipped = 0
        for r in voz_labeled:
            # Get label: prefer auto_label, fallback to label
            label_raw = r.get("auto_label") or r.get("label")
            lb = normalize_label(label_raw)

            if not lb:
                voz_skipped += 1
                continue

            rid = r.get("id")
            if rid in seen_ids:
                voz_skipped += 1
                continue

            # Inject cascade from raw file
            if rid in cascade_lookup:
                r["cascade"] = cascade_lookup[rid]
            else:
                r["cascade"] = []

            # Inject media_url if missing
            if not r.get("media_url") and rid in media_lookup:
                r["media_url"] = media_lookup[rid]

            # Assign random split (80/10/10)
            import random
            rnd = random.random()
            if rnd < 0.8:
                r["split"] = "train"
            elif rnd < 0.9:
                r["split"] = "val"
            else:
                r["split"] = "test"

            r["label_binary"] = lb
            r["label"] = lb
            r["source_dataset"] = "voz"
            r.setdefault("raw_text", r.get("clean_text", ""))

            seen_ids.add(rid)
            all_records.append(r)
            label_counts[lb] += 1
            source_counts["Voz"] += 1
            voz_added += 1

        print(f"  [Voz]     {voz_added} added, {voz_skipped} skipped (no label or duplicate)")
    else:
        print(f"  [Voz]     File not found: {voz_labeled_path}")

    # ─────────────────────────────────────────────
    # 3. NEWS (All REAL - báo chính thống)
    # ─────────────────────────────────────────────
    news_path = Path(args.news)
    if news_path.exists():
        news = read_jsonl(news_path)
        for r in news:
            rid = r.get("id")
            if rid in seen_ids:
                continue

            r["label_binary"] = "REAL"
            r["label"] = "REAL"
            r.setdefault("cascade", [])
            r["source_dataset"] = "news"
            r.setdefault("raw_text", r.get("raw_text", ""))

            # Assign random split (80/10/10)
            import random
            rnd = random.random()
            if rnd < 0.8:
                r["split"] = "train"
            elif rnd < 0.9:
                r["split"] = "val"
            else:
                r["split"] = "test"

            seen_ids.add(rid)
            all_records.append(r)
            label_counts["REAL"] += 1
            source_counts["News"] += 1
        print(f"  [News]    {source_counts.get('News', 0)} records loaded (all REAL)")
    else:
        print(f"  [News]    File not found: {news_path}")

    # ─────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_records, output_path)

    print()
    print("=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    print(f"  Output:  {output_path}")
    print(f"  Total:   {len(all_records)} records")
    print(f"  Sources: {dict(source_counts)}")
    print(f"  Labels:  {dict(label_counts)}")
    print(f"  REAL:    {label_counts.get('REAL', 0)}")
    print(f"  FAKE:    {label_counts.get('FAKE', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
