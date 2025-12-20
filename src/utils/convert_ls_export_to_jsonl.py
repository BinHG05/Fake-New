import json
import sys
import os
from pathlib import Path
import argparse

def convert_ls_export_to_jsonl(input_path, output_path):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ File không tồn tại: {input_path}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("❌ Lỗi: File input không phải là JSON hợp lệ.")
        return

    print(f"🔄 Đang xử lý {len(data)} tasks từ {input_file.name}...")
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for task in data:
            # Label Studio export structure:
            # task = { "id": 1, "data": { ... }, "annotations": [ ... ] }
            
            # 1. Lấy dữ liệu gốc từ trường 'data'
            # Nếu không có 'data', dùng chính task đó (tùy format export)
            record = task.get('data', task).copy()
            
            # Xóa các trường thừa nếu record lấy từ root task (tránh trùng)
            if 'data' in task:
                # Nếu 'data' tách riêng, record đã sạch
                pass
            else:
                # Nếu export dạng flattened, cần bỏ các field của LS
                for key in ['annotations', 'predictions', 'id', 'created_at', 'updated_at', 'project']:
                    record.pop(key, None)

            # 2. Trích xuất NHÃN từ annotations
            ls_label = None
            ls_notes = None
            
            annotations = task.get('annotations', [])
            if annotations:
                # Lấy annotation cuối cùng (mới nhất)
                last_annotation = annotations[-1]
                result = last_annotation.get('result', [])
                
                for r in result:
                    # Tìm nhãn (hỗ trợ nhiều tên commonly used)
                    if r.get('from_name') in ['label', 'label_fine', 'choice']:
                        choices = r.get('value', {}).get('choices', [])
                        if choices:
                            ls_label = choices[0]
                    
                    # Tìm ghi chú (from_name='notes') nếu có
                    if r.get('from_name') == 'notes':
                        ls_notes = r.get('value', {}).get('text', [None])[0]

            # 3. Ghi thông tin nhãn vào record
            if ls_label:
                record['label'] = ls_label  # Ghi đè hoặc thêm mới
                record['manual_label'] = True
            else:
                record['manual_label'] = False # Chưa gán nhãn
            
            if ls_notes:
                record['notes'] = ls_notes

            # Giữ lại ID của Label Studio để đối chiếu
            record['ls_id'] = task.get('id')

            # 4. Ghi ra dòng JSONL
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
            
    print(f"✅ Đã chuyển đổi thành công {count} dòng.")
    print(f"💾 File lưu tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio JSON export to JSONL')
    parser.add_argument('input', help='Path to Label Studio JSON export file')
    parser.add_argument('output', nargs='?', help='Path to output JSONL file (default: same name .jsonl)')
    
    args = parser.parse_args()
    
    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        # Tự động tạo tên file output: input.json -> input.jsonl
        output_path = str(Path(input_path).with_suffix('.jsonl'))
    
    convert_ls_export_to_jsonl(input_path, output_path)

if __name__ == "__main__":
    main()
