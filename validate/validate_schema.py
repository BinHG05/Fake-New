# validate_schema.py
import json
import os
import time
import logging
import sys # <-- THÊM MỚI ĐỂ SỬA LỖI WINDOWS
from collections import Counter
from datetime import datetime

from jsonschema import validate, ValidationError
# Giả định schema_definitions.py đã được tạo ở bước 1
from schema_definitions import CORE_SCHEMA, EXTENDED_SCHEMA 

# Cấu hình Logging
LOG_FILE = 'validation_report.log'
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s: %(message)s',
                    handlers=[
                        # Thêm encoding='utf-8' cho FileHandler
                        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
                        # Dùng sys.stdout để ép UTF-8 trên console Windows
                        logging.StreamHandler(sys.stdout) 
                    ])
logger = logging.getLogger(__name__)

# --- CÁC HÀM KIỂM TRA CHÍNH ---

def validate_jsonl_file(file_path, schema, schema_name):
    """
    Thực hiện kiểm tra cấu trúc (jsonschema) và logic nghiệp vụ trong một lần quét (One-Pass).
    """
    logger.info(f"\n--- Bắt đầu kiểm tra {schema_name} ({file_path}) ---")
    
    validation_summary = {
        'struct_errors': 0,
        'logic_errors': 0,
        'duplicate_ids': [],
        'post_ids': [],
        'user_ids': set()
    }
    
    if not os.path.exists(file_path):
        logger.error(f"LỖI: Không tìm thấy file tại đường dẫn: {file_path}")
        validation_summary['struct_errors'] += 1
        return validation_summary

    total_records = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f): 
            if not line.strip(): continue
            total_records += 1
            
            try:
                data = json.loads(line)
                
                # 1. Kiểm tra Cấu trúc và Kiểu dữ liệu (JSONSchema)
                validate(instance=data, schema=schema)
                
                # Thu thập ID
                if 'id' in data: validation_summary['post_ids'].append(data['id'])
                if 'user_id' in data: validation_summary['user_ids'].add(data['user_id'])
                
                # 2. Kiểm tra Logic Nghiệp vụ
                if schema_name == "EXTENDED_SCHEMA":
                    img_info = data.get('image_info', {})
                    text_content = data.get('clean_text', '')
                    
                    # 2.1. Kiểm tra Quy tắc Cấu trúc cụ thể (image_size phải là [224, 224])
                    if img_info.get('image_size') not in ([224, 224], None):
                        logger.error(f"LỖI LOGIC Dòng {line_number + 1}: image_size không phải [224, 224].")
                        validation_summary['logic_errors'] += 1
                        
                    # 2.2. Kiểm tra consistency của video/keyframe
                    is_video = img_info.get('is_video', False)
                    keyframe_paths = img_info.get('keyframe_paths', [])
                    if is_video and not keyframe_paths:
                        logger.error(f"LỖI LOGIC Dòng {line_number + 1}: is_video=True nhưng keyframe_paths trống.")
                        validation_summary['logic_errors'] += 1
                    elif not is_video and keyframe_paths:
                        logger.error(f"LỖI LOGIC Dòng {line_number + 1}: is_video=False nhưng keyframe_paths có dữ liệu.")
                        validation_summary['logic_errors'] += 1

                    # 2.3. Kiểm tra clean_text consistency (ĐÃ SỬA LỖI LOGIC isupper - Lỗi 1)
                    
                    # Logic 1: Kiểm tra độ dài
                    is_length_valid = len(text_content) >= 5 and len(text_content) <= 5000
                    
                    # Logic 2: Kiểm tra chữ hoa (any upper case char)
                    has_uppercase = any(c.isupper() for c in text_content if c.isalpha())
                    
                    if not is_length_valid:
                        logger.error(f"LỖI LOGIC Dòng {line_number + 1}: clean_text không nhất quán (quá ngắn/quá dài).")
                        validation_summary['logic_errors'] += 1
                    elif has_uppercase:
                        logger.error(f"LỖI LOGIC Dòng {line_number + 1}: clean_text chứa ký tự chữ hoa (chưa lowercase).")
                        validation_summary['logic_errors'] += 1
                        
                    # 2.4. Kiểm tra timestamp logic
                    current_epoch = int(time.time())
                    if data.get('timestamp') and data.get('timestamp') > current_epoch:
                        logger.error(f"LỖI LOGIC Dòng {line_number + 1}: timestamp tương lai.")
                        validation_summary['logic_errors'] += 1

                    # 2.5. Kiểm tra media_url <-> is_video 
                    if data.get('media_url') and not img_info.get('processed_path'):
                        logger.warning(f"CẢNH BÁO Dòng {line_number + 1}: Có media_url thô nhưng thiếu processed_path.")

                
            except json.JSONDecodeError:
                logger.error(f"LỖI CÚ PHÁP Dòng {line_number + 1}: JSON không hợp lệ.")
                validation_summary['struct_errors'] += 1
            except ValidationError as e:
                logger.error(f"LỖI SCHEMA Dòng {line_number + 1} ({e.path}): {e.message}")
                validation_summary['struct_errors'] += 1
                
    # 3. Kiểm tra Tính Duy Nhất (Logic Cấu trúc cuối cùng)
    id_counts = Counter(validation_summary['post_ids'])
    validation_summary['duplicate_ids'] = [id for id, count in id_counts.items() if count > 1]
    validation_summary['struct_errors'] += len(validation_summary['duplicate_ids'])
    
    if validation_summary['duplicate_ids']:
        logger.error(f"LỖI CẤU TRÚC: {len(validation_summary['duplicate_ids'])} ID bị trùng lặp.")
        
    logger.info(f"Hoàn tất kiểm tra {total_records} bản ghi. Tổng lỗi: {validation_summary['struct_errors'] + validation_summary['logic_errors']}.")
    return validation_summary 

# --- HÀM CHẠY CHÍNH VÀ TỔNG HỢP ---

def run_validation(raw_path: str, processed_path: str):
    """
    Run validation on specified files.
    
    Args:
        raw_path: Path to raw JSONL file (CORE_SCHEMA)
        processed_path: Path to processed JSONL file (EXTENDED_SCHEMA)
    """
    # 1. KIỂM TRA ĐẦU VÀO (CORE SCHEMA)
    core_results = validate_jsonl_file(raw_path, CORE_SCHEMA, "CORE_SCHEMA")

    # 2. KIỂM TRA ĐẦU RA (EXTENDED SCHEMA)
    extended_results = validate_jsonl_file(processed_path, EXTENDED_SCHEMA, "EXTENDED_SCHEMA")

    # 3. TỔNG KẾT LỖI
    total_errors = core_results['struct_errors'] + core_results['logic_errors'] + \
                   extended_results['struct_errors'] + extended_results['logic_errors']
    
    logger.info("\n" + "="*70)
    logger.info("                         ✨ BÁO CÁO TỔNG KẾT VALIDATION ✨")
    logger.info("="*70)
    logger.info(f"TỔNG SỐ LỖI PHÁT HIỆN: {total_errors}")
    logger.info(f"Report chi tiết đã được ghi vào file: {LOG_FILE}")
    
    logger.info("\n--- CHI TIẾT TÓM TẮT ---")
    logger.info(f"CORE SCHEMA (A) - Lỗi Cấu trúc: {core_results['struct_errors']}")
    logger.info(f"EXTENDED SCHEMA (B/C) - Lỗi Cấu trúc: {extended_results['struct_errors']}")
    logger.info(f"EXTENDED SCHEMA (B/C) - Lỗi Logic Nghiệp vụ: {extended_results['logic_errors']}")
    
    if total_errors > 0:
        logger.error("⚠ HÀNH ĐỘNG: CẦN YÊU CẦU CÁC THÀNH VIÊN SỬA DỮ LIỆU. Vui lòng xem log.")
    else:
        logger.info("👍 DỮ LIỆU ĐẠT CHUẨN. Có thể tiếp tục Gán nhãn/Xây dựng Graph.")
    
    return total_errors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Fakeddit data against schemas')
    parser.add_argument(
        '--raw',
        default='data/01_raw/Fakeddit/Fakeddit_pilot_processed_200.jsonl',
        help='Path to raw JSONL file (CORE_SCHEMA)'
    )
    parser.add_argument(
        '--processed',
        default='data/03_clean/Fakeddit/train.jsonl',
        help='Path to processed JSONL file (EXTENDED_SCHEMA)'
    )
    
    args = parser.parse_args()
    run_validation(args.raw, args.processed)