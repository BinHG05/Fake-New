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

def convert_path_for_label_studio(relative_path: str, docker_mode: bool = False) -> str:
    """
    Chuyển đổi đường dẫn tương đối sang format Label Studio local storage.
    """
    if not relative_path:
        return ""
    
    # Xử lý các tiền tố tuyệt đối để đưa về đường dẫn tương đối dự án
    # (Trường hợp dữ liệu đầu vào đã bị gắn nhãn tuyệt đối hoặc format sai)
    clean_path = relative_path.replace("\\", "/")
    
    # Danh sách các tiền tố cần loại bỏ để đưa về tương đối từ project root
    prefixes_to_strip = [
        "D:/NCKH_Project/Project/",
        "/label-studio/project/",
        "/data/local-files/?d="
    ]
    
    for prefix in prefixes_to_strip:
        if clean_path.startswith(prefix):
            clean_path = clean_path[len(prefix):]
        if clean_path.startswith(prefix.lower()): # Đề phòng case-insensitive
            clean_path = clean_path[len(prefix):]
    
    # Trim leading slash nếu có
    clean_path = clean_path.lstrip("/")

    
    if docker_mode:
        # Trong Docker, Label Studio sẽ kết hợp DOCUMENT_ROOT với đường dẫn sau ?d=
        # Nếu ta set DOCUMENT_ROOT=/label-studio/project, thì path ở đây phải là TƯƠNG ĐỐI
        # Ví dụ: data/02_processed/images/...
        return f"/data/local-files/?d={clean_path}"
    else:
        # Chế độ Windows Local: Dùng đường dẫn tuyệt đối là chắc chắn nhất
        abs_path = os.path.join(project_root, relative_path)
        abs_path = os.path.abspath(abs_path).replace("\\", "/")
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
                    
                    # Chuyển đổi đường dẫn ảnh nếu có image_info
                    if convert_paths and 'image_info' in record:
                        processed_path = record['image_info'].get('processed_path', '')
                        if processed_path:
                            # 1. Update trực tiếp vào image_info để khớp với mẫu của USER
                            ls_path = convert_path_for_label_studio(processed_path, docker_mode)
                            record['image_info']['processed_path'] = ls_path
                            
                            # 2. Thêm trường image ở root để Label Studio dễ nhận diện mặc định
                            record['image'] = ls_path
                    
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
        python src/utils/convert_to_ls_json.py --input data/03_clean/Fakeddit/train.jsonl --output data/03_clean/Fakeddit/train_for_ls.json
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JSONL to JSON for Label Studio')
    parser.add_argument('--input', help='Path to input JSONL file or directory')
    parser.add_argument('--output', help='Path to output JSON file or output directory')
    parser.add_argument('--batch-name', help='Batch name to process splits (train, val, test) inside a folder')
    parser.add_argument('--docker', action='store_true', 
                        help='Use Docker mode (assumes mount data:/label-studio/data)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CONVERT JSONL TO JSON FOR LABEL STUDIO")
    print("=" * 60)
    print(f"Mode: {'Docker' if args.docker else 'Local'}")
    print()

    # Determine files to process
    tasks = []
    
    if args.batch_name:
        # Process splits inside a batch folder in 03_clean
        batch_dir = Path("data/03_clean/Fakeddit") / args.batch_name
        if not batch_dir.exists():
            print(f"❌ Error: Batch directory not found: {batch_dir}")
            return
            
        # Check for nested Fakeddit folder (created by processor)
        if (batch_dir / "Fakeddit").exists():
            batch_dir = batch_dir / "Fakeddit"
            
        for split in ['train', 'val', 'test']:
            input_file = batch_dir / f"{split}.jsonl"
            output_file = batch_dir / f"{split}_for_ls.json"
            if input_file.exists():
                tasks.append((str(input_file), str(output_file)))
    elif args.input and args.output:
        # Specific input/output
        tasks.append((args.input, args.output))
    else:
        # Default fallback (original logic for backward compatibility if no args)
        if not args.input:
            print("🔄 No specific input provided, checking default splits in data/03_clean/Fakeddit/")
            for split in ['train', 'val', 'test']:
                input_file = f"data/03_clean/Fakeddit/{split}.jsonl"
                output_file = f"data/03_clean/Fakeddit/{split}_for_ls.json"
                if Path(input_file).exists():
                    tasks.append((input_file, output_file))
        else:
            print("❌ Error: Please provide both --input and --output, or use --batch-name")
            return

    if not tasks:
        print("⚠️ No files found to process.")
        return

    for input_jsonl, output_json in tasks:
        print(f"🔄 Processing: {Path(input_jsonl).name} -> {Path(output_json).name}")
        convert_jsonl_to_json(input_jsonl, output_json, docker_mode=args.docker)
        print("-" * 40)
    
    print()
    print("=" * 60)
    if args.docker:
        print("HƯỚNG DẪN DOCKER (LINUX CONTAINER):")
        print("1. Chạy Docker mount project folder vào /label-studio/project:")
        print("   docker run -d -p 8080:8080 \\")
        print("     -v D:/NCKH_Project/Project:/label-studio/project \\")
        print("     --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\")
        print("     --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/project \\")
        print("     --name label-studio heartexlabs/label-studio:latest")
        print()
        print("2. Import file JSON vừa tạo vào Label Studio.")
        print("   Đường dẫn ảnh trong file sẽ là: /data/local-files/?d=data/02_processed/images/...")
    else:
        print("HƯỚNG DẪN LOCAL (WINDOWS CMD):")
        print("1. Set môi trường:")
        print("   set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"   set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={project_root}")
        print("2. Chạy: label-studio")
        print("3. Import file JSON vừa tạo.")
    print("=" * 60)


if __name__ == "__main__":
    main()