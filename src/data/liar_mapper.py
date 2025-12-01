import pandas as pd
import json
import os
import time
import random
import logging
import zipfile
from datetime import datetime

# IMPORT THƯ VIỆN KAGGLE API
from kaggle.api.kaggle_api_extended import KaggleApi

# -----------------------------------------------------
# 1. LOGGING SETUP (Production)
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -----------------------------------------------------
# 2. CẤU TRÚC DỮ LIỆU GỐC CỦA LIAR
# -----------------------------------------------------
COLUMN_NAMES = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state_info", "party_affiliation",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

DATASET_NAME = "doanquanvietnamca/liar-dataset"

# -----------------------------------------------------
# 3. HÀM VALIDATE SCHEMA (Import từ nhóm D)
# -----------------------------------------------------
def validate_record(record):
    required_fields = ["id", "timestamp", "label", "raw_text", "user_id"]

    for field in required_fields:
        if record.get(field) in [None, ""]:
            return False, f"Missing required field: {field}"

    if len(record["raw_text"].strip()) < 3:
        return False, "Raw text too short."

    return True, None


# -----------------------------------------------------
# 4. DOWNLOAD + EXTRACT
# -----------------------------------------------------
def download_and_extract_liar(download_dir="data/01_raw"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    logging.info(f"Bắt đầu tải dataset từ Kaggle: {DATASET_NAME}")

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_NAME, path=download_dir, unzip=False)
    except Exception as e:
        logging.error("Không thể tải dataset từ Kaggle. Kiểm tra API key.")
        logging.error(e)
        return None

    zip_path = os.path.join(download_dir, "liar-dataset.zip")

    if not os.path.exists(zip_path):
        logging.error("Không tìm thấy file ZIP.")
        return None

    # Giải nén toàn bộ
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    os.remove(zip_path)
    logging.info("Giải nén hoàn tất.")

    # trả về danh sách file tsv
    extracted = [
        os.path.join(download_dir, f)
        for f in os.listdir(download_dir)
        if f.endswith(".tsv")       
    ]

    logging.info(f"Đã tìm thấy {len(extracted)} file TSV: {extracted}")

    return extracted


# -----------------------------------------------------
# 5. CHUẨN HÓA + ÁNH XẠ LIAR → CORE SCHEMA
# -----------------------------------------------------
def map_liar_to_core_schema(
        file_path,
        output_dir="data/01_raw",
        num_samples=500,
        save_parquet=True,
        timestamp_min_year=2010,
        timestamp_max_year=2020
):
    if not os.path.exists(file_path):
        logging.error("Không tìm thấy file train.tsv.")
        return

    logging.info(f"Đang đọc file: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        keep_default_na=False,
        quoting=3
    )

    # Lọc câu không hợp lệ
    df = df[df["statement"].apply(lambda x: isinstance(x, str) and len(x.strip()) >= 3)]
    logging.info(f"Sau khi lọc câu trống: còn {len(df)} mẫu.")

    # Giới hạn số mẫu
    if num_samples != -1:
        df = df.head(num_samples)

    mapped_records = []
    seen_ids = set()

    # Timestamp RANGE
    start_ts = int(datetime(timestamp_min_year, 1, 1).timestamp())
    end_ts = int(datetime(timestamp_max_year, 12, 31).timestamp())

    for _, row in df.iterrows():
        record_id = f"LIAR_{row['id']}"
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)

        # Random timestamp hợp lý
        rand_timestamp = random.randint(start_ts, end_ts)

        core_record = {
            "id": record_id,
            "timestamp": rand_timestamp,
            "timestamp_status": "RANDOMIZED_RANGE",

            "label": row["label"],
            "raw_text": row["statement"],
            "media_url": "NONE",
            "user_id": row["speaker"],
            "retweet_count": 0,

            # Metadata
            "subject": row["subject"],
            "statement_context": row["context"],
            "user_job_title": row["speaker_job"],
            "user_state_info": row["state_info"],
            "user_party": row["party_affiliation"],

            # FULL credit-history
            "user_credit_history": {
                "barely_true": int(row["barely_true_counts"]),
                "false": int(row["false_counts"]),
                "half_true": int(row["half_true_counts"]),
                "mostly_true": int(row["mostly_true_counts"]),
                "pants_on_fire": int(row["pants_on_fire_counts"])
            }
        }

        # Validate trước khi lưu
        ok, err = validate_record(core_record)
        if ok:
            mapped_records.append(core_record)
        else:
            logging.warning(f"BỎ QUA {record_id} — {err}")

    # OUTPUT FILE
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, f"liar_mapped_{len(mapped_records)}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in mapped_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"Tạo JSONL thành công: {jsonl_path}")

    # Optional: Lưu Parquet cho Big Data
    if save_parquet:
        parquet_path = jsonl_path.replace(".jsonl", ".parquet")
        pd.DataFrame(mapped_records).to_parquet(parquet_path, index=False)
        logging.info(f"Đã tạo file Parquet: {parquet_path}")

    return mapped_records


# -----------------------------------------------------
# 6. MAIN (Cho thành viên A chạy)
# -----------------------------------------------------
def main():
    tsv_files = download_and_extract_liar()
    if not tsv_files:
        return

    for tsv_file in tsv_files:
        split_name = os.path.basename(tsv_file).replace(".tsv", "")
        logging.info(f"=== PROCESSING SPLIT: {split_name} ===")

        map_liar_to_core_schema(
            file_path=tsv_file,
            output_dir=f"data/01_raw/liar_{split_name}",
            num_samples=-1  # dùng toàn bộ dataset
        )



if __name__ == "__main__":
    main()
