"""
Enrich labeled_master.jsonl with Reddit comment trees and write reddit_enriched_data.jsonl.

This script is a clean, CLI-friendly replacement for the older crawler_enrich.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.reddit_crawler import RedditCrawler


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    existing_ids = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                existing_ids.add(str(json.loads(line)["id"]))
            except Exception:
                continue
    return existing_ids


def build_permalink(post: dict) -> str:
    permalink = (
        post.get("metadata", {}).get("permalink")
        or post.get("permalink")
        or f"/comments/{post.get('id', '')}/"
    )
    if not permalink.endswith("/"):
        permalink += "/"
    return permalink


def enrich_records(input_path: Path, output_path: Path, delay_seconds: float, limit: int | None) -> None:
    crawler = RedditCrawler(debug=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_records = read_jsonl(input_path)
    done_ids = get_existing_ids(output_path)

    print(f"Input records: {len(source_records)}")
    print(f"Already enriched: {len(done_ids)}")

    processed = 0
    skipped = 0

    for index, post in enumerate(source_records, start=1):
        post_id = str(post.get("id", "")).strip()
        if not post_id:
            continue
        if post_id in done_ids:
            skipped += 1
            continue
        if limit is not None and processed >= limit:
            break

        permalink = build_permalink(post)
        print(f"[{index}/{len(source_records)}] Fetching cascade for {post_id} ...")

        try:
            cascade_data = crawler.fetch_comments(permalink)
        except Exception as exc:
            print(f"  Failed for {post_id}: {exc}")
            cascade_data = []

        enriched_record = dict(post)
        enriched_record["cascade"] = cascade_data
        enriched_record["metadata_enrich"] = {
            "enriched_at": int(time.time()),
            "comment_count_fetched": len(cascade_data),
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")

        processed += 1
        print(f"  Saved {post_id} with {len(cascade_data)} comments")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("=" * 60)
    print("ENRICH COMPLETE")
    print(f"Newly processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich master data with Reddit comment trees")
    parser.add_argument(
        "--input",
        default="data/03_clean/Fakeddit/labeled_master.jsonl",
        help="Input master JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/reddit_enriched_data.jsonl",
        help="Output enriched JSONL",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between posts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of new posts to enrich",
    )
    args = parser.parse_args()

    enrich_records(
        input_path=Path(args.input),
        output_path=Path(args.output),
        delay_seconds=args.delay,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
