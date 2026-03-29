"""
Package Data for Team Sharing
=============================
Dong goi cac file du lieu can thiet (khong co tren GitHub) thanh file ZIP
de gui cho thanh vien nhom chay web demo.

Usage:
    python package_data.py                  # Goi muc toi thieu (~30 MB)
    python package_data.py --full           # Goi day du (bao gom graphs + checkpoints)
    python package_data.py --full --no-large-checkpoints  # Bo checkpoint > 100MB
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent

# -- Danh sach file can dong goi --

MINIMAL_FILES = [
    # Du lieu da lam giau (de web hien thi stats + build graphs)
    "data/reddit_enriched_data.jsonl",
    # Du lieu tho (de web dem so bai viet)
    "data/01_raw/reddit/reddit_realtime_data.jsonl",
    # Visualization HTML
    "data/cascade_visualization.html",
]

GRAPH_DIRS = [
    "data/processed_graphs",               # Text-only graphs (768d)
    "data/processed_graphs_multimodal",     # Multimodal graphs (1280d)
]

CHECKPOINT_DIR = "models/checkpoints"

LARGE_CHECKPOINT_THRESHOLD = 100 * 1024 * 1024  # 100 MB


def get_size_str(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f} MB"
    else:
        return f"{size_bytes / (1024*1024*1024):.2f} GB"


def collect_files(full=False, no_large_checkpoints=False):
    """Thu thap danh sach file can dong goi."""
    files = []

    # Luon them minimal files
    for rel_path in MINIMAL_FILES:
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists():
            files.append(rel_path)
        else:
            print(f"  [SKIP] Khong tim thay: {rel_path}")

    if not full:
        return files

    # Them graph directories
    for graph_dir in GRAPH_DIRS:
        dir_path = PROJECT_ROOT / graph_dir
        if dir_path.exists():
            for pt_file in dir_path.glob("*.pt"):
                files.append(str(pt_file.relative_to(PROJECT_ROOT)))
        else:
            print(f"  [SKIP] Khong tim thay thu muc: {graph_dir}")

    # Them checkpoints
    ckpt_dir = PROJECT_ROOT / CHECKPOINT_DIR
    if ckpt_dir.exists():
        for ckpt_file in ckpt_dir.iterdir():
            if ckpt_file.suffix == ".pt":
                size = ckpt_file.stat().st_size
                if no_large_checkpoints and size > LARGE_CHECKPOINT_THRESHOLD:
                    print(f"  [SKIP] Checkpoint qua lon ({get_size_str(size)}): {ckpt_file.name}")
                    continue
                files.append(str(ckpt_file.relative_to(PROJECT_ROOT)))

    return files


def main():
    parser = argparse.ArgumentParser(description="Dong goi du lieu cho team")
    parser.add_argument("--full", action="store_true", help="Goi day du (graphs + checkpoints)")
    parser.add_argument("--no-large-checkpoints", action="store_true",
                        help="Bo qua checkpoint > 100MB")
    parser.add_argument("--output", type=str, default=None, help="Ten file ZIP dau ra")
    args = parser.parse_args()

    mode = "full" if args.full else "minimal"
    timestamp = datetime.now().strftime("%Y%m%d")

    if args.output:
        zip_name = args.output
    else:
        zip_name = f"fakenews_data_{mode}_{timestamp}.zip"

    print(f"\n{'='*60}")
    print(f"  DONG GOI DU LIEU - Che do: {mode.upper()}")
    print(f"{'='*60}\n")

    # Thu thap file
    files = collect_files(full=args.full, no_large_checkpoints=args.no_large_checkpoints)

    if not files:
        print("Khong tim thay file nao de dong goi!")
        sys.exit(1)

    # Tinh tong dung luong
    total_size = 0
    print(f"Cac file se duoc dong goi:\n")
    for rel_path in sorted(files):
        abs_path = PROJECT_ROOT / rel_path
        size = abs_path.stat().st_size
        total_size += size
        print(f"  {get_size_str(size):>10}  {rel_path}")

    print(f"\n  {'_'*40}")
    print(f"  {'Tong cong:':>10}  {len(files)} files, {get_size_str(total_size)}")
    print()

    # Xac nhan
    confirm = input(f"Tao file {zip_name}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Da huy.")
        return

    # Tao ZIP
    print(f"\nDang nen...")
    zip_path = PROJECT_ROOT / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, rel_path in enumerate(files):
            abs_path = PROJECT_ROOT / rel_path
            zf.write(abs_path, rel_path)
            print(f"  [{i+1}/{len(files)}] {rel_path}")

    final_size = zip_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"  HOAN THANH!")
    print(f"  File: {zip_path}")
    print(f"  Dung luong: {get_size_str(final_size)}")
    print(f"  Ty le nen: {final_size/total_size*100:.1f}%")
    print(f"{'='*60}")
    print(f"\nGui file nay kem huong dan trong demo-website/README.md cho thanh vien nhom.")


if __name__ == "__main__":
    main()
