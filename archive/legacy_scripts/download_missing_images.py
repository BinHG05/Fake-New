"""
Download missing Fakeddit images using REAL URLs from raw dataset.

Cross-references labeled_master_binary.jsonl (for missing image paths)
with dataset_Fakeddit_Processed.jsonl (for real media_url).

Usage:
    python scripts/download_missing_images.py
"""

import json
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = PROJECT_ROOT / "data" / "03_clean" / "Fakeddit" / "labeled_master_binary.jsonl"
RAW_PATH = PROJECT_ROOT / "data" / "01_raw" / "Fakeddit" / "dataset_Fakeddit_Processed.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

TARGET_SIZE = (224, 224)
MAX_WORKERS = 8
TIMEOUT = 20


def resolve_image_path(raw_path: str) -> Path:
    p = raw_path.replace("\\", "/")
    if "?d=" in p:
        p = p.split("?d=")[-1]
    p = p.lstrip("/")
    if p.startswith("data/"):
        return PROJECT_ROOT / p
    else:
        return PROJECT_ROOT / "data" / p


def download_and_resize(url: str, save_path: Path) -> bool:
    """Download image from URL, resize to 224x224, save as JPEG."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "image" in content_type or "octet-stream" in content_type or len(resp.content) > 1000:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(save_path), "JPEG", quality=95)
                return True
    except Exception:
        pass
    return False


def main():
    print("=" * 65)
    print("  DOWNLOAD MISSING FAKEDDIT IMAGES (using real URLs)")
    print("=" * 65)

    # ── Step 1: Build URL lookup from raw dataset ──
    print(f"\n1) Loading URL database from: {RAW_PATH.name}")
    url_map = {}  # id -> media_url
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id", "")
            url = rec.get("media_url", "")
            if rid and url:
                url_map[rid] = url

    print(f"   Loaded {len(url_map)} URLs")

    # ── Step 2: Find missing images ──
    print(f"\n2) Scanning labeled dataset: {LABELED_PATH.name}")
    missing = []
    already_exists = 0
    no_path = 0
    no_url = 0

    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw_p = rec.get("image_info", {}).get("processed_path", "")
            if not raw_p:
                no_path += 1
                continue

            resolved = resolve_image_path(raw_p)
            if resolved.exists():
                already_exists += 1
            else:
                post_id = resolved.stem
                url = url_map.get(post_id, "")
                if url:
                    missing.append({
                        "post_id": post_id,
                        "save_path": resolved,
                        "url": url,
                    })
                else:
                    no_url += 1

    print(f"   Already exists: {already_exists}")
    print(f"   No path:        {no_path}")
    print(f"   No URL found:   {no_url}")
    print(f"   To download:    {len(missing)}")

    if not missing:
        print("\nNothing to download!")
        return

    # ── Step 3: Download ──
    print(f"\n3) Downloading {len(missing)} images ({MAX_WORKERS} workers)...")
    print("-" * 65)

    success = 0
    failed = 0
    failed_ids = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(download_and_resize, item["url"], item["save_path"]): item
            for item in missing
        }

        for i, future in enumerate(as_completed(future_map), 1):
            item = future_map[future]
            try:
                ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    failed_ids.append(item["post_id"])
            except Exception:
                failed += 1
                failed_ids.append(item["post_id"])

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(missing) - i) / rate if rate > 0 else 0
            print(
                f"\r  [{i}/{len(missing)}] "
                f"OK={success} FAIL={failed} "
                f"| {rate:.1f} img/s | ETA: {eta:.0f}s",
                end="", flush=True,
            )

    elapsed_total = time.time() - t0
    print()
    print("-" * 65)
    print(f"\nDONE in {elapsed_total:.0f}s")
    print(f"  Downloaded: {success}/{len(missing)} ({success/len(missing)*100:.1f}%)")
    print(f"  Failed:     {failed}")

    if failed_ids:
        fail_log = PROJECT_ROOT / "scripts" / "failed_downloads.txt"
        with open(fail_log, "w") as f:
            for pid in failed_ids:
                f.write(pid + "\n")
        print(f"  Failed IDs saved to: {fail_log}")

    print(f"\n  Total images now: {already_exists + success}/{already_exists + success + failed + no_path}")

    print("\n" + "=" * 65)
    if success > 0:
        print("Next steps:")
        print("  1) python src/features/clip_text_extractor_v2.py")
        print("  2) python src/training/train_compare_v1_v2.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
