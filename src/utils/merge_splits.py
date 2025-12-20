"""
Merge train/val/test JSONL files into single file with 'split' field.
For unified graph construction.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def merge_jsonl_files(
    train_path: str,
    val_path: str,
    test_path: str,
    output_path: str
) -> Dict[str, int]:
    """
    Merge train/val/test JSONL into single file with 'split' field.
    
    Args:
        train_path: Path to train JSONL
        val_path: Path to val JSONL
        test_path: Path to test JSONL
        output_path: Path to output merged JSONL
        
    Returns:
        Dict with counts per split
    """
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        
        # Process each split
        for split_name, file_path in [
            ('train', train_path),
            ('val', val_path),
            ('test', test_path)
        ]:
            path = Path(file_path)
            if not path.exists():
                print(f"⚠️ File không tồn tại, bỏ qua: {file_path}")
                continue
            
            print(f"📖 Đang đọc {split_name}: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        # Add split field
                        record['split'] = split_name
                        f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                        counts[split_name] += 1
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON error: {e}")
                        continue
    
    return counts


def main():
    parser = argparse.ArgumentParser(
        description='Merge train/val/test JSONL files into single file'
    )
    parser.add_argument(
        '--train', 
        default='labels_storage/export/train_done.jsonl',
        help='Path to train JSONL'
    )
    parser.add_argument(
        '--val',
        default='labels_storage/export/val_done.jsonl', 
        help='Path to val JSONL'
    )
    parser.add_argument(
        '--test',
        default='labels_storage/export/test_done.jsonl',
        help='Path to test JSONL'
    )
    parser.add_argument(
        '--output',
        default='data/04_graph/merged_data.jsonl',
        help='Path to output merged JSONL'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MERGE JSONL FILES FOR GRAPH CONSTRUCTION")
    print("=" * 60)
    
    counts = merge_jsonl_files(
        args.train,
        args.val, 
        args.test,
        args.output
    )
    
    total = sum(counts.values())
    print()
    print("=" * 60)
    print(f"✅ Merged {total} records:")
    print(f"   Train: {counts['train']}")
    print(f"   Val:   {counts['val']}")
    print(f"   Test:  {counts['test']}")
    print(f"💾 Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
