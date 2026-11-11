"""
Reddit Data Processing Pipeline
================================
Giống batch_pipeline.py của Fakeddit nhưng dùng cho dữ liệu Reddit.
Dùng CHUNG các script xử lý để output có cùng format → gộp chung khi train.

2 chế độ:
    1. AUTO (mặc định): Tự phát hiện data mới chưa xử lý
    2. MANUAL (--start --count): Chỉ định batch thủ công như Fakeddit

Luồng xử lý:
    Extract batch → Image Processing → Text Processing → Validation → Label Studio

Chạy:
    python src/utils/reddit_pipeline.py                          # Auto: xử lý tất cả data mới
    python src/utils/reddit_pipeline.py --start 0 --count 50     # Manual: lấy 50 bản ghi từ vị trí 0
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_INPUT = os.path.join(PROJECT_ROOT, "data", "01_raw", "reddit", "reddit_realtime_data.jsonl")
BATCH_DIR = os.path.join(PROJECT_ROOT, "data", "01_raw", "reddit", "batches")
PROCESSED_LOG = os.path.join(PROJECT_ROOT, "data", "01_raw", "reddit", "processed_tracker.json")


def run_command(command: list, description: str):
    """Chạy lệnh và in trạng thái."""
    print(f"\n🚀 {description}...")
    print(f"   Command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True)
        print(f"✅ {description} hoàn tất.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} thất bại (code {e.returncode})")
        return False


def load_tracker():
    """Đọc tracker: danh sách ID đã xử lý + số batch đã chạy."""
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_ids": [], "run_count": 0}


def save_tracker(tracker):
    """Lưu tracker."""
    os.makedirs(os.path.dirname(PROCESSED_LOG), exist_ok=True)
    with open(PROCESSED_LOG, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)


def read_all_records(input_file):
    """Đọc toàn bộ records từ JSONL."""
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_batch(records, output_file):
    """Lưu batch records ra file JSONL."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"   📄 Đã trích xuất {len(records)} bản ghi → {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Reddit Data Processing Pipeline')
    parser.add_argument('--input', default=RAW_INPUT,
                        help='File JSONL nguồn từ reddit_crawler.py')
    parser.add_argument('--start', type=int, default=None,
                        help='(Manual) Vị trí bắt đầu (0-based)')
    parser.add_argument('--count', type=int, default=None,
                        help='(Manual) Số lượng bản ghi trong batch')

    args = parser.parse_args()
    is_manual = args.start is not None and args.count is not None

    # ====================================================
    # KIỂM TRA FILE INPUT
    # ====================================================
    if not os.path.exists(args.input):
        print(f"❌ Không tìm thấy file: {args.input}")
        print(f"→ Chạy 'python src/data/reddit_crawler.py' trước!")
        sys.exit(1)

    all_records = read_all_records(args.input)
    print("=" * 60)
    print("🔄 REDDIT DATA PROCESSING PIPELINE")
    print("=" * 60)
    print(f"   📊 Tổng records trong file nguồn: {len(all_records)}")

    # ====================================================
    # TRÍCH XUẤT BATCH
    # ====================================================
    tracker = load_tracker()
    processed_ids = set(tracker.get("processed_ids", []))
    run_count = tracker.get("run_count", 0) + 1

    if is_manual:
        # ---- CHẾ ĐỘ MANUAL (như Fakeddit --start --count) ----
        end = min(args.start + args.count, len(all_records))
        batch_records = all_records[args.start:end]
        batch_name = f"reddit_{args.start}_{end}"
        print(f"   📋 Chế độ: MANUAL (start={args.start}, count={args.count})")
        print(f"   📋 Batch:  {batch_name} ({len(batch_records)} bản ghi)")
    else:
        # ---- CHẾ ĐỘ AUTO (chỉ lấy data mới) ----
        new_records = [r for r in all_records if r.get('id') not in processed_ids]

        if not new_records:
            print("\n✅ Không có dữ liệu mới để xử lý!")
            print(f"   Đã xử lý: {len(processed_ids)} / {len(all_records)} bản ghi")
            print("→ Chạy 'python src/data/reddit_crawler.py' để crawl thêm data.")
            return

        batch_records = new_records
        batch_name = f"reddit_run_{run_count:03d}"
        print(f"   📋 Chế độ: AUTO")
        print(f"   📋 Đã xử lý trước đó: {len(processed_ids)} bản ghi")
        print(f"   📋 Data mới: {len(new_records)} bản ghi → {batch_name}")

    print("=" * 60)

    # Lưu batch ra file riêng
    batch_file = os.path.join(BATCH_DIR, f"{batch_name}.jsonl")
    extract_batch(batch_records, batch_file)

    # ====================================================
    # SETUP PATHS (giống batch_pipeline.py)
    # ====================================================
    image_processed_file = "data/02_processed/dataset_output.jsonl"
    processed_val_file = f"data/03_clean/Reddit/{batch_name}/Reddit/train.jsonl"
    output_clean_dir = f"data/03_clean/Reddit/{batch_name}"

    # Xóa file tạm cũ để cô lập batch
    if os.path.exists(image_processed_file):
        print(f"\n🗑️ Xóa file tạm cũ: {image_processed_file}")
        os.remove(image_processed_file)

    # ====================================================
    # BƯỚC 1: Image Processing (dùng chung script)
    # ====================================================
    image_cmd = [
        "python", "src/data/fakeddit_preprocessor_image.py",
        "--input", batch_file,
        "--batch-name", batch_name,
        "--dataset-name", "Reddit"
    ]
    if not run_command(image_cmd, f"Bước 1/4: Xử lý ảnh ({len(batch_records)} bản ghi)"):
        sys.exit(1)

    # ====================================================
    # BƯỚC 2: Text Processing (dùng chung script)
    # ====================================================
    text_cmd = [
        "python", "src/data/fakeddit_process_text.py",
        "--input", image_processed_file,
        "--batch-name", batch_name,
        "--dataset-name", "Reddit"
    ]
    if not run_command(text_cmd, "Bước 2/4: Xử lý text (Clean & Features)"):
        sys.exit(1)

    # ====================================================
    # BƯỚC 3: Validation (dùng chung validate_schema.py)
    # ====================================================
    validate_cmd = [
        "python", "validate/validate_schema.py",
        "--raw", batch_file,
        "--processed", processed_val_file
    ]
    if not run_command(validate_cmd, "Bước 3/4: Validate Schema"):
        print("⚠️ Validation có lỗi nhưng pipeline vẫn tiếp tục.")

    # ====================================================
    # BƯỚC 4: Convert cho Label Studio
    # ====================================================
    try:
        if os.path.exists(processed_val_file):
            ls_output_file = os.path.join(output_clean_dir, "reddit_for_ls.json")
            os.makedirs(output_clean_dir, exist_ok=True)
            convert_cmd = [
                "python", "src/utils/convert_to_ls_json.py",
                "--input", processed_val_file,
                "--output", ls_output_file,
                "--docker"
            ]
            run_command(convert_cmd, "Bước 4/4: Convert sang Label Studio Format")
        else:
            print(f"⚠️ Không tìm thấy {processed_val_file}")
    except Exception as e:
        print(f"⚠️ Lỗi ở bước LS conversion: {e}")

    # ====================================================
    # CẬP NHẬT TRACKER
    # ====================================================
    new_ids = [r.get('id') for r in batch_records if r.get('id')]
    tracker["processed_ids"] = list(processed_ids | set(new_ids))
    tracker["run_count"] = run_count
    tracker["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tracker["last_batch"] = batch_name
    tracker["last_batch_size"] = len(batch_records)
    save_tracker(tracker)

    # ====================================================
    # TỔNG KẾT
    # ====================================================
    print("\n" + "=" * 60)
    print(f"🎉 PIPELINE HOÀN TẤT — {batch_name}")
    print("=" * 60)
    print(f"   📊 Đã xử lý:          {len(batch_records)} bản ghi")
    print(f"   📊 Tổng đã xử lý:     {len(tracker['processed_ids'])} / {len(all_records)}")
    print(f"   📊 Còn lại chưa xử lý: {len(all_records) - len(tracker['processed_ids'])}")
    print(f"   📁 Batch file:         {batch_file}")
    print(f"   📁 Data sạch:          data/03_clean/Reddit/{batch_name}/Reddit/")
    print(f"   📄 File Label Studio:  {output_clean_dir}/reddit_for_ls.json")
    print("=" * 60)
    print()
    print("👉 Cách dùng tiếp:")
    print("   • Chạy lại pipeline:  python src/utils/reddit_pipeline.py")
    print("     → Tự xử lý data mới (nếu có)")
    print("   • Chia batch thủ công: python src/utils/reddit_pipeline.py --start 50 --count 25")
    print("   • Crawl thêm data:   python src/data/reddit_crawler.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
