"""
Utility script to convert JSONL files (line-delimited JSON) 
into a single JSON file (JSON Array) for reliable import into Label Studio.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def convert_jsonl_to_json(input_path: str, output_path: str) -> int:
    """
    Đọc file JSONL, gom các bản ghi thành một mảng JSON Array, và ghi ra file JSON.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"❌ Lỗi: Không tìm thấy file input tại {input_path}")
        return 0

    print(f"Đang đọc file JSONL: {input_path}")
    data_array: List[Dict] = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    data_array.append(record)
                except json.JSONDecodeError:
                    print(f"⚠️ Cảnh báo: Lỗi JSON tại dòng {line_num}. Bỏ qua bản ghi.")
                    continue

    if not data_array:
        print("❌ Lỗi: Không có bản ghi hợp lệ nào được tìm thấy.")
        return 0

    print(f"✓ Đã đọc {len(data_array)} bản ghi.")
    
    # Ghi ra file JSON Array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_array, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Chuyển đổi thành công. Lưu tại: {output_file}")
    return len(data_array)

def main():
    # Cấu hình đường dẫn cho Fakeddit Pilot
    INPUT_JSONL = "data/03_clean/Fakeddit/train.jsonl"
    
    # ĐÃ CẬP NHẬT: Đặt file output vào data/03_clean/Fakeddit/
    OUTPUT_JSON = "data/03_clean/Fakeddit/train_for_ls.json" 

    # Đảm bảo thư mục output tồn tại
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    
    convert_jsonl_to_json(INPUT_JSONL, OUTPUT_JSON)

if __name__ == "__main__":
    main()