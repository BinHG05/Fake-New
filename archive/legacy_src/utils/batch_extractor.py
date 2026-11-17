"""
Batch Extractor for Fakeddit Dataset.
Splits raw JSONL into chunks of N samples for incremental labeling.

Usage:
    python src/utils/batch_extractor.py --start 200 --count 200 --output batch2
"""

import json
import argparse
from pathlib import Path


def extract_batch(
    input_file: str,
    start_idx: int,
    count: int,
    output_dir: str
) -> str:
    """
    Extract a batch of records from JSONL file.
    
    Args:
        input_file: Path to source JSONL
        start_idx: Starting index (0-based)
        count: Number of records to extract
        output_dir: Output directory
        
    Returns:
        Path to output file
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read all lines
    print(f"📖 Reading from: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total = len(lines)
    print(f"   Total records: {total}")
    
    # Validate range
    if start_idx >= total:
        print(f"❌ Error: start_idx ({start_idx}) >= total ({total})")
        return None
    
    end_idx = min(start_idx + count, total)
    actual_count = end_idx - start_idx
    
    # Extract batch
    batch_lines = lines[start_idx:end_idx]
    
    # Create output filename: Fakeddit_{start}_{end}.jsonl
    output_filename = f"Fakeddit_{start_idx}_{end_idx}.jsonl"
    output_file = output_path / output_filename
    
    print(f"📤 Extracting records [{start_idx} - {end_idx}) ({actual_count} records)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in batch_lines:
            f.write(line)
    
    print(f"✅ Saved to: {output_file}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Extract batch from Fakeddit JSONL')
    parser.add_argument(
        '--input',
        default='data/01_raw/Fakeddit/dataset_Fakeddit_Processed.jsonl',
        help='Path to source JSONL file'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=200,
        help='Starting index (0-based). Default: 200 (skip first 200 already processed)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=200,
        help='Number of records to extract per batch'
    )
    parser.add_argument(
        '--output',
        default='data/01_raw/Fakeddit/batches',
        help='Output directory for batch files'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FAKEDDIT BATCH EXTRACTOR")
    print("=" * 60)
    
    result = extract_batch(
        args.input,
        args.start,
        args.count,
        args.output
    )
    
    if result:
        print()
        print("=" * 60)
        print(f"✅ Done! File created: {result}")
        print(f"   Next batch: --start {args.start + args.count}")
        print("=" * 60)


if __name__ == "__main__":
    main()
