"""
Image Preprocessing Module for Fake News Detection
VERSION 3: Dataset chung + File riêng cho mỗi lần run
Author: Member C
"""

import sys
import os
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("IMAGE PREPROCESSOR V3 - SHARED + INDIVIDUAL OUTPUTS")
print("=" * 60)
print(f"Python: {sys.version}")
print()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("Importing libraries...")

import json
import logging
from typing import Dict, List, Optional, Tuple
import requests
from io import BytesIO
import re

print("  ✓ Basic imports")

from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("  ✓ PIL")

from tqdm import tqdm
print("  ✓ tqdm")

print()
print("All imports successful! Starting processor...")
print()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OutputFileManager:
    """Quản lý file output: file chung (append) + file riêng (auto-increment)"""
    
    @staticmethod
    def get_next_filename(base_path: str, extension: str = ".jsonl") -> str:
        """
        Tìm tên file tiếp theo có sẵn (không bị ghi đè)
        
        Args:
            base_path: Đường dẫn cơ bản (ví dụ: "data/processed/dataset")
            extension: Phần mở rộng file
            
        Returns:
            Đường dẫn file mới với số thứ tự
        """
        base_path = Path(base_path)
        base_dir = base_path.parent
        base_name = base_path.stem
        
        # Tạo thư mục nếu chưa có
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra file gốc
        original_file = base_dir / f"{base_name}{extension}"
        if not original_file.exists():
            return str(original_file)
        
        # Tìm số thứ tự tiếp theo
        counter = 1
        while True:
            new_file = base_dir / f"{base_name}_{counter:03d}{extension}"
            if not new_file.exists():
                return str(new_file)
            counter += 1
    
    @staticmethod
    def append_to_shared_file(shared_file: str, records: List[Dict]) -> None:
        """
        Append records vào file chung (không ghi đè)
        
        Args:
            shared_file: Đường dẫn file chung
            records: Danh sách records cần thêm
        """
        shared_path = Path(shared_file)
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append mode
        with open(shared_path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    @staticmethod
    def count_existing_records(file_path: str) -> int:
        """Đếm số records đã có trong file"""
        if not os.path.exists(file_path):
            return 0
        
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count


class ImageProcessor:
    """Process images for fake news detection (NO VIDEO SUPPORT, NO DISTORTION)"""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        output_base_dir: str = "data/02_processed",
        timeout: int = 15,
        max_samples: int = 200,
        padding_color: Tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Initialize the processor
        
        Args:
            target_size: Target image size (width, height)
            output_base_dir: Base directory for processed files
            timeout: Timeout for downloading images (seconds)
            max_samples: Maximum number of samples to process
            padding_color: RGB color for padding (default: black)
        """
        self.target_size = target_size
        self.output_base_dir = Path(output_base_dir)
        self.timeout = timeout
        self.max_samples = max_samples
        self.padding_color = padding_color
        
        # Tạo timestamp cho batch này
        self.batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_id = f"batch_{self.batch_timestamp}"
        
        # Statistics
        self.stats = {
            "total_records": 0,
            "skipped_no_media": 0,
            "skipped_video": 0,
            "images_success": 0,
            "download_failed": 0,
            "processing_failed": 0,
            "batch_id": self.batch_id
        }
    
    def setup_output_directories(self, dataset_name: str):
        """
        Thiết lập thư mục output
        
        Args:
            dataset_name: Tên dataset (ví dụ: "Fakeddit")
        """
        # Thư mục images cho batch này
        self.images_dir = self.output_base_dir / "images" / f"{dataset_name}_{self.batch_timestamp}"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Images directory: {self.images_dir}")
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(
                url, 
                timeout=self.timeout, 
                headers=headers,
                verify=False
            )
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            return img
            
        except Exception:
            return None
    
    def resize_image_with_padding(
        self,
        img: Image.Image,
        convert_rgb: bool = True
    ) -> Optional[Image.Image]:
        """Resize image to target size WITHOUT distortion using padding"""
        try:
            if convert_rgb and img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_width, original_height = img.size
            target_width, target_height = self.target_size
            
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if original_ratio > target_ratio:
                new_width = target_width
                new_height = int(target_width / original_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * original_ratio)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            new_img = Image.new('RGB', self.target_size, self.padding_color)
            
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            new_img.paste(img_resized, (paste_x, paste_y))
            
            return new_img
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return None
    
    def process_image(
        self,
        media_url: str,
        post_id: str,
        use_padding: bool = True
    ) -> Optional[Dict]:
        """Process a single image"""
        try:
            img = self._download_image(media_url)
            
            if img is None:
                self.stats["download_failed"] += 1
                return None
            
            if use_padding:
                img_resized = self.resize_image_with_padding(img)
            else:
                img_resized = ImageOps.fit(
                    img, 
                    self.target_size, 
                    Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5)
                )
                
            if img_resized is None:
                self.stats["processing_failed"] += 1
                return None
            
            # Save với tên có timestamp
            output_filename = f"{post_id}.jpg"
            output_path = self.images_dir / output_filename
            img_resized.save(output_path, "JPEG", quality=90)
            
            # Tạo relative path
            relative_path = str(output_path.relative_to(Path(".")))
            
            self.stats["images_success"] += 1
            
            return {
                "processed_path": relative_path,
                "image_size": list(self.target_size),
                "is_video": False,
                "keyframe_paths": [],
                "processing_method": "padding" if use_padding else "crop",
                "batch_id": self.batch_id
            }
            
        except Exception as e:
            logger.error(f"Error processing image for {post_id}: {e}")
            self.stats["processing_failed"] += 1
            return None
    
    def is_video_url(self, url: str) -> bool:
        """Check if URL points to a video file"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        return any(url.lower().endswith(ext) for ext in video_extensions)
    
    def process_batch(
        self,
        input_jsonl: str,
        shared_output: str,
        individual_output_base: str,
        dataset_name: str = "Fakeddit",
        use_padding: bool = True
    ) -> Tuple[str, str]:
        """
        Process a batch of posts from JSONL file
        
        Args:
            input_jsonl: Path to input JSONL file (raw data)
            shared_output: Path to shared dataset file (APPEND mode)
            individual_output_base: Base path for individual file (AUTO-INCREMENT)
            dataset_name: Dataset name for organizing files
            use_padding: If True, use padding; if False, use crop
            
        Returns:
            Tuple of (shared_file_path, individual_file_path)
        """
        # Setup output directories
        self.setup_output_directories(dataset_name)
        
        # Tìm tên file individual không bị ghi đè
        individual_output = OutputFileManager.get_next_filename(individual_output_base)
        
        # Đếm số records đã có trong file chung
        existing_count = OutputFileManager.count_existing_records(shared_output)
        
        print(f"Input file:        {input_jsonl}")
        print(f"Shared output:     {shared_output} (existing: {existing_count} records)")
        print(f"Individual output: {individual_output}")
        print(f"Max samples:       {self.max_samples}")
        print(f"Method:            {'Padding' if use_padding else 'Center crop'}")
        print(f"Batch ID:          {self.batch_id}")
        print()
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        processed_records = []
        
        try:
            # Read input file
            print("Reading input file...")
            with open(input_jsonl, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            lines = lines[:self.max_samples]
            
            print(f"Total records to process: {len(lines)}")
            print("=" * 60)
            print()
            
            # Process each line
            for line in tqdm(lines, desc="Processing images", ncols=80):
                try:
                    record = json.loads(line.strip())
                    self.stats["total_records"] += 1
                    
                    if 'media_url' not in record or not record['media_url']:
                        self.stats["skipped_no_media"] += 1
                        continue
                    
                    post_id = record.get('id', f"post_{self.stats['total_records']}")
                    media_url = record['media_url']
                    
                    if self.is_video_url(media_url):
                        self.stats["skipped_video"] += 1
                        continue
                    
                    image_info = self.process_image(media_url, post_id, use_padding=use_padding)
                    
                    if image_info is not None:
                        record['image_info'] = image_info
                        processed_records.append(record)
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON line")
                    continue
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
                    continue
            
            print()
            print("=" * 60)
            print("Saving processed records...")
            
            # 1. APPEND vào file chung
            print(f"Appending {len(processed_records)} records to shared file...")
            OutputFileManager.append_to_shared_file(shared_output, processed_records)
            new_total = existing_count + len(processed_records)
            print(f"✓ Shared file now has {new_total} records")
            
            # 2. Lưu file riêng cho batch này
            print(f"Saving individual batch file...")
            with open(individual_output, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"✓ Individual file saved: {individual_output}")
            
            # Save metadata
            metadata = {
                "batch_id": self.batch_id,
                "shared_file": shared_output,
                "individual_file": individual_output,
                "images_dir": str(self.images_dir),
                "existing_records_before": existing_count,
                "new_records_added": len(processed_records),
                "total_records_after": new_total,
                "processing_stats": self.stats,
                "config": {
                    "target_size": self.target_size,
                    "max_samples": self.max_samples,
                    "use_padding": use_padding
                }
            }
            
            metadata_file = Path(individual_output).parent / f"metadata_{self.batch_timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Metadata: {metadata_file}")
            print()
            self._print_statistics()
            
            return shared_output, individual_output
            
        except Exception as e:
            print(f"\nERROR in batch processing: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_statistics(self) -> None:
        """Print processing statistics"""
        print("=" * 60)
        print("PROCESSING STATISTICS:")
        print("=" * 60)
        print(f"Batch ID:               {self.batch_id}")
        print(f"Total records read:     {self.stats['total_records']}")
        print(f"Skipped (no media):     {self.stats['skipped_no_media']}")
        print(f"Skipped (video):        {self.stats['skipped_video']}")
        print(f"Images processed:       {self.stats['images_success']}")
        print(f"Download failed:        {self.stats['download_failed']}")
        print(f"Processing failed:      {self.stats['processing_failed']}")
        
        total_attempted = (self.stats['total_records'] - 
                          self.stats['skipped_no_media'] - 
                          self.stats['skipped_video'])
        
        if total_attempted > 0:
            success_rate = (self.stats['images_success'] / total_attempted) * 100
            print(f"Success rate:           {success_rate:.2f}%")
        print("=" * 60)


def main():
    """Main function to run the processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images for Fakeddit dataset')
    parser.add_argument(
        '--input',
        default='data/01_raw/Fakeddit/Fakeddit_pilot_processed_200.jsonl',
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--batch-name',
        default=None,
        help='Batch name for output files (e.g., "200_400"). If not provided, derived from input filename.'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("IMAGE PROCESSOR V3 - SHARED + INDIVIDUAL OUTPUTS")
    print("VERSION: No Distortion (Padding Method)")
    print("FEATURE: Shared dataset (append) + Individual files (no overwrite)")
    print("=" * 60)
    print()
    
    # Configuration
    INPUT_FILE = args.input
    
    # Derive batch name from input file if not provided
    if args.batch_name:
        batch_name = args.batch_name
    else:
        # Extract from filename like "Fakeddit_200_400.jsonl" -> "200_400"
        import re
        match = re.search(r'Fakeddit_(\d+_\d+)', os.path.basename(INPUT_FILE))
        if match:
            batch_name = match.group(1)
        else:
            batch_name = "batch"
    
    # File chung (APPEND mode - không ghi đè)
    SHARED_OUTPUT = "data/02_processed/dataset_output.jsonl"
    
    # File riêng cho batch này (AUTO-INCREMENT - không ghi đè)
    INDIVIDUAL_OUTPUT_BASE = f"data/02_processed/Fakeddit/dataset_Fakeddit_{batch_name}"
    
    DATASET_NAME = "Fakeddit"
    
    USE_PADDING = True
    MAX_SAMPLES = args.max_samples
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        print()
        return

    
    file_size = os.path.getsize(INPUT_FILE)
    print(f"✓ Input file found: {INPUT_FILE}")
    print(f"  File size: {file_size:,} bytes")
    print()
    
    # Check existing shared file
    if os.path.exists(SHARED_OUTPUT):
        existing_count = OutputFileManager.count_existing_records(SHARED_OUTPUT)
        print(f"✓ Shared dataset exists: {SHARED_OUTPUT}")
        print(f"  Existing records: {existing_count}")
        print(f"  Mode: APPEND (will add new records)")
    else:
        print(f"✓ Shared dataset will be created: {SHARED_OUTPUT}")
        print(f"  Mode: CREATE NEW")
    print()
    
    # Initialize processor
    print("Initializing processor...")
    processor = ImageProcessor(
        target_size=(224, 224),
        output_base_dir="data/02_processed",
        timeout=15,
        max_samples=MAX_SAMPLES,
        padding_color=(0, 0, 0)
    )
    print("✓ Processor initialized")
    print()
    
    # Process batch
    try:
        shared_file, individual_file = processor.process_batch(
            input_jsonl=INPUT_FILE,
            shared_output=SHARED_OUTPUT,
            individual_output_base=INDIVIDUAL_OUTPUT_BASE,
            dataset_name=DATASET_NAME,
            use_padding=USE_PADDING
        )
        
        print()
        print("=" * 60)
        print("✓ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("OUTPUT FILES:")
        print(f"1. Shared dataset:    {shared_file}")
        print(f"   (All batches combined, safe to append)")
        print(f"2. Individual batch:  {individual_file}")
        print(f"   (This batch only, for reference)")
        print(f"3. Images directory:  {processor.images_dir}")
        print()
        print("Next steps:")
        print("1. Review the output files and images")
        print("2. Run fakeddit_process_text.py to clean the data and split")
        print("3. Use DVC to track the processed data")
        print()
        print("DVC commands:")
        print(f"  dvc add {shared_file}")
        print(f"  dvc add {individual_file}")
        print(f"  dvc add {processor.images_dir}")
        print(f"  git add *.dvc .gitignore")
        print(f"  git commit -m 'Add processed data batch {processor.batch_id}'")
        print(f"  dvc push")
        print()
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠  INTERRUPTED BY USER")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ PROCESSING FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")