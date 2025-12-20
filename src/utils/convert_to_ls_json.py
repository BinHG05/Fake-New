"""
Utility script to convert JSONL files (line-delimited JSON) 
into a single JSON file (JSON Array) for reliable import into Label Studio.

UPDATED: Chuyển đổi đường dẫn ảnh sang format Label Studio local storage.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def convert_path_for_label_studio(relative_path: str, project_root: str, docker_mode: bool = False) -> str:
    """
    Chuyển đổi đường dẫn tương đối sang format Label Studio local storage.
    
    Args:
        relative_path: Đường dẫn tương đối từ project root (e.g., data/02_processed/images/...)
        project_root: Đường dẫn project root (local)
        docker_mode: Nếu True, tạo path cho Docker 
                     (path tương đối từ thư mục data/ vì mount data:/label-studio/data)
    
    Label Studio cần format: /data/local-files/?d=<path_relative_to_DOCUMENT_ROOT>
    """
    if not relative_path:
        return ""
    
    # Chuyển backslash thành forward slash
    relative_path = relative_path.replace("\\", "/")
    
    if docker_mode:
        # Docker mode: mount là data:/label-studio/data
        # DOCUMENT_ROOT=/label-studio/data
        # Nên path cần là phần sau "data/" 
        # Ví dụ: data/02_processed/images/... -> 02_processed/images/...
        if relative_path.startswith("data/"):
            # Bỏ prefix "data/" vì nó đã được mount vào /label-studio/data
            path_from_root = relative_path[5:]  # Remove "data/"
        else:
            path_from_root = relative_path
        
        return f"/data/local-files/?d={path_from_root}"
    else:
        # Local mode: dùng đường dẫn tuyệt đối
        abs_path = os.path.join(project_root, relative_path)
        abs_path = os.path.abspath(abs_path)
        abs_path = abs_path.replace("\\", "/")
        
        return f"/data/local-files/?d={abs_path}"


def convert_jsonl_to_json(input_path: str, output_path: str, convert_paths: bool = True, docker_mode: bool = False) -> int:
    """
    Đọc file JSONL, gom các bản ghi thành một mảng JSON Array, và ghi ra file JSON.
    
    Args:
        input_path: Đường dẫn file JSONL input
        output_path: Đường dẫn file JSON output  
        convert_paths: Nếu True, chuyển đổi đường dẫn ảnh sang format Label Studio
        docker_mode: Nếu True, tạo path cho Docker (mount data:/label-studio/data)
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"❌ Lỗi: Không tìm thấy file input tại {input_path}")
        return 0

    print(f"Đang đọc file JSONL: {input_path}")
    print(f"Convert paths for Label Studio: {convert_paths}")
    print(f"Docker mode: {docker_mode}")
    
    data_array: List[Dict] = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    
                    # Chuyển đổi đường dẫn ảnh nếu cần
                    if convert_paths and 'image_info' in record:
                        processed_path = record['image_info'].get('processed_path', '')
                        if processed_path:
                            record['image_info']['processed_path'] = convert_path_for_label_studio(
                                processed_path, project_root, docker_mode
                            )
                            # Thêm trường image cho Label Studio
                            record['image'] = record['image_info']['processed_path']
                    
                    data_array.append(record)
                except json.JSONDecodeError:
                    print(f"⚠️ Cảnh báo: Lỗi JSON tại dòng {line_num}. Bỏ qua bản ghi.")
                    continue

    if not data_array:
        print("❌ Lỗi: Không có bản ghi hợp lệ nào được tìm thấy.")
        return 0

    print(f"✓ Đã đọc {len(data_array)} bản ghi.")
    
    if convert_paths and data_array:
        print(f"  Ví dụ đường dẫn ảnh sau convert:")
        print(f"  {data_array[0].get('image', 'N/A')}")
    
    # Ghi ra file JSON Array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_array, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Chuyển đổi thành công. Lưu tại: {output_file}")
    return len(data_array)

def main():
    """
    Main function - convert JSONL to JSON for Label Studio
    
    Usage:
        python convert_to_ls_json.py              # Local mode
        python convert_to_ls_json.py --docker     # Docker mode
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSONL to JSON for Label Studio')
    parser.add_argument('--docker', action='store_true', 
                        help='Use Docker mode (assumes mount data:/label-studio/data)')
    args = parser.parse_args()
    
    # Cấu hình đường dẫn cho Fakeddit Pilot
    splits = ['train', 'val', 'test']
    
    print("=" * 60)
    print("CONVERT JSONL TO JSON FOR LABEL STUDIO")
    print("=" * 60)
    print(f"Mode: {'Docker' if args.docker else 'Local'}")
    print()
    
    for split in splits:
        input_jsonl = f"data/03_clean/Fakeddit/{split}.jsonl"
        output_json = f"data/03_clean/Fakeddit/{split}_for_ls.json"
        
        # Đảm bảo thư mục output tồn tại
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra file input có tồn tại không
        if Path(input_jsonl).exists():
            print(f"🔄 Đang xử lý conversion cho split: {split}")
            convert_jsonl_to_json(input_jsonl, output_json, docker_mode=args.docker)
            print("-" * 40)
        else:
            print(f"⚠️ Bỏ qua split {split}: File không tồn tại ({input_jsonl})")
    
    print()
    print("=" * 60)
    if args.docker:
        print("HƯỚNG DẪN DOCKER:")
        print("1. Mount thư mục data vào container:")
        print("   docker run -v D:\\NCKH_Project\\Project\\data:/label-studio/data ...")
        print("2. Set DOCUMENT_ROOT=/label-studio/data")
        print("3. Cấu hình Local Storage với path: /label-studio/data/02_processed/images")
        print("4. Import file <split>_for_ls.json vào Label Studio")
        print()
        print("Ví dụ URL ảnh: /data/local-files/?d=02_processed/images/Fakeddit_.../abc.jpg")
    else:
        print("HƯỚNG DẪN LOCAL:")
        print("1. Set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print("2. Set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=D:/NCKH_Project/Project")
        print("3. Import file <split>_for_ls.json vào Label Studio")
    print("=" * 60)

if __name__ == "__main__":
    main()