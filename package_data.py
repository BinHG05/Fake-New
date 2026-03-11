\"\"\"
Package Data for Team Sharing
=============================
Đóng gói các file dữ liệu cần thiết (không có trên GitHub) thành file ZIP
để gửi cho thành viên nhóm chạy web demo.

Usage:
    python package_data.py                  # Gói mức tối thiểu (~30 MB)
    python package_data.py --full           # Gói đầy đủ (bao gồm graphs + checkpoints)
    python package_data.py --full --no-large-checkpoints  # Bỏ checkpoint > 100MB
\"\"\"

import os
import sys
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Danh sách file cần đóng gói ──

MINIMAL_FILES = [
    # Dữ liệu đã làm giàu (để web hiển thị stats + build graphs)
    "data/reddit_enriched_data.jsonl",
    # Dữ liệu thô (để web đếm số bài viết)
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
    \"\"\"Thu thập danh sách file cần đóng gói.\"\"\"
    files = []

    # Luôn thêm minimal files
    for rel_path in MINIMAL_FILES:
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists():
            files.append(rel_path)
        else:
            print(f"  [SKIP] Không tìm thấy: {rel_path}")

    if not full:
        return files

    # Thêm graph directories
    for graph_dir in GRAPH_DIRS:
        dir_path = PROJECT_ROOT / graph_dir
        if dir_path.exists():
            for pt_file in dir_path.glob("*.pt"):
                files.append(str(pt_file.relative_to(PROJECT_ROOT)))
        else:
            print(f"  [SKIP] Không tìm thấy thư mục: {graph_dir}")

    # Thêm checkpoints
    ckpt_dir = PROJECT_ROOT / CHECKPOINT_DIR
    if ckpt_dir.exists():
        for ckpt_file in ckpt_dir.iterdir():
            if ckpt_file.suffix == ".pt":
                size = ckpt_file.stat().st_size
                if no_large_checkpoints and size > LARGE_CHECKPOINT_THRESHOLD:
                    print(f"  [SKIP] Checkpoint quá lớn ({get_size_str(size)}): {ckpt_file.name}")
                    continue
                files.append(str(ckpt_file.relative_to(PROJECT_ROOT)))

    return files


def main():
    parser = argparse.ArgumentParser(description="Đóng gói dữ liệu cho team")
    parser.add_argument("--full", action="store_true", help="Gói đầy đủ (graphs + checkpoints)")
    parser.add_argument("--no-large-checkpoints", action="store_true",
                        help="Bỏ qua checkpoint > 100MB")
    parser.add_argument("--output", type=str, default=None, help="Tên file ZIP đầu ra")
    args = parser.parse_args()

    mode = "full" if args.full else "minimal"
    timestamp = datetime.now().strftime("%Y%m%d")

    if args.output:
        zip_name = args.output
    else:
        zip_name = f"fakenews_data_{mode}_{timestamp}.zip"

    print(f"\n{'='*60}")
    print(f"  ĐÓNG GÓI DỮ LIỆU - Chế độ: {mode.upper()}")
    print(f"{'='*60}\n")

    # Thu thập file
    files = collect_files(full=args.full, no_large_checkpoints=args.no_large_checkpoints)

    if not files:
        print("Không tìm thấy file nào để đóng gói!")
        sys.exit(1)

    # Tính tổng dung lượng
    total_size = 0
    print(f"Các file sẽ được đóng gói:\n")
    for rel_path in sorted(files):
        abs_path = PROJECT_ROOT / rel_path
        size = abs_path.stat().st_size
        total_size += size
        print(f"  {get_size_str(size):>10}  {rel_path}")

    print(f"\n  {'─'*40}")
    print(f"  {'Tổng cộng:':>10}  {len(files)} files, {get_size_str(total_size)}")
    print()

    # Xác nhận
    confirm = input(f"Tạo file {zip_name}? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Đã hủy.")
        return

    # Tạo ZIP
    print(f"\nĐang nén...")
    zip_path = PROJECT_ROOT / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, rel_path in enumerate(files):
            abs_path = PROJECT_ROOT / rel_path
            zf.write(abs_path, rel_path)
            print(f"  [{i+1}/{len(files)}] {rel_path}")

    final_size = zip_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"  HOÀN THÀNH!")
    print(f"  File: {zip_path}")
    print(f"  Dung lượng: {get_size_str(final_size)}")
    print(f"  Tỷ lệ nén: {final_size/total_size*100:.1f}%")
    print(f"{'='*60}")
    print(f"\nGửi file này kèm hướng dẫn trong demo-website/README.md cho thành viên nhóm.")


if __name__ == "__main__":
    main()
