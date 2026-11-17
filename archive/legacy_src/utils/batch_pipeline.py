"""
Batch Processing Pipeline for Fakeddit Dataset.
Automates 4 steps: Extraction -> Image Processing -> Text Processing -> Validation.

Usage:
    python src/utils/batch_pipeline.py --start 200 --count 200
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command: list, description: str):
    """Run a shell command and print status."""
    print(f"\n🚀 {description}...")
    print(f"Command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True)
        print(f"✅ {description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Automated Batch Processing Pipeline')
    parser.add_argument('--start', type=int, required=True, help='Starting index (0-based)')
    parser.add_argument('--count', type=int, default=200, help='Number of samples in batch')
    parser.add_argument('--input_raw', default='data/01_raw/Fakeddit/dataset_Fakeddit_Processed.jsonl', help='Path to master raw file')
    
    args = parser.parse_args()
    
    # 0. Setup Batch Info
    end = args.start + args.count
    batch_name = f"batch_{args.start}_{end}"
    raw_batch_file = f"data/01_raw/Fakeddit/batches/Fakeddit_{args.start}_{end}.jsonl"
    image_processed_file = "data/02_processed/dataset_output.jsonl"
    output_clean_dir = f"data/03_clean/Fakeddit/{batch_name}"
    
    print("=" * 60)
    print(f"🎬 STARTING PIPELINE FOR {batch_name.upper()}")
    print("=" * 60)

    # Step 0: Ensure dataset_output.jsonl is removed to isolate this batch
    if os.path.exists(image_processed_file):
        print(f"🗑️ Removing old {image_processed_file} for isolation...")
        os.remove(image_processed_file)

    # Step 1: Extraction
    extract_cmd = [
        "python", "src/utils/batch_extractor.py",
        "--input", args.input_raw,
        "--start", str(args.start),
        "--count", str(args.count)
    ]
    if not run_command(extract_cmd, "Extracting Raw Batch"):
        sys.exit(1)

    # Step 2: Image Processing
    image_cmd = [
        "python", "src/data/fakeddit_preprocessor_image.py",
        "--input", raw_batch_file
    ]
    if not run_command(image_cmd, "Image Preprocessing (Download & Resize)"):
        sys.exit(1)

    # Step 3: Text Processing
    text_cmd = [
        "python", "src/data/fakeddit_process_text.py",
        "--input", image_processed_file,
        "--batch-name", batch_name
    ]
    if not run_command(text_cmd, "Text Preprocessing & Splitting"):
        sys.exit(1)

    # Step 4: Validation
    # We use the train.jsonl in the batch folder for validation
    processed_val_file = f"data/03_clean/Fakeddit/{batch_name}/Fakeddit/train.jsonl"
    validate_cmd = [
        "python", "validate/validate_schema.py",
        "--raw", raw_batch_file,
        "--processed", processed_val_file
    ]
    if not run_command(validate_cmd, "Validating Batch Schema"):
        print("⚠️ Validation reported issues, but pipeline finished. Check logs.")
    # Step 5: Convert to JSON for Label Studio (using existing script)
    # train.jsonl -> train_for_ls.json
    try:
        if os.path.exists(processed_val_file):
            ls_output_file = f"{output_clean_dir}/train_for_ls.json"
            convert_cmd = [
                "python", "src/utils/convert_to_ls_json.py",
                "--input", processed_val_file,
                "--output", ls_output_file,
                "--docker" # Assuming Docker is used as per guide, or we can make this configurable
            ]
            run_command(convert_cmd, "Converting to Label Studio Format")
        else:
            print(f"⚠️ Could not find {processed_val_file} to convert for Label Studio.")
    except Exception as e:
        print(f"⚠️ Error in LS conversion step: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 BATCH {batch_name.upper()} PROCESSED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📂 Folder for team: {output_clean_dir}")
    print(f"📄 LS File: {output_clean_dir}/train_for_ls.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
