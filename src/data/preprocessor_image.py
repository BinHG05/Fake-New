"""
Image Preprocessing Module for Fake News Detection
IMAGE ONLY VERSION - No distortion with padding
Author: Member C
"""

import sys
print("=" * 60)
print("IMAGE PREPROCESSOR - NO DISTORTION VERSION")
print("=" * 60)
print(f"Python: {sys.version}")
print()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

print("Importing libraries...")

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from io import BytesIO

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
        
        # Create output directories
        self.images_dir = self.output_base_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_records": 0,
            "skipped_no_media": 0,
            "skipped_video": 0,
            "images_success": 0,
            "download_failed": 0,
            "processing_failed": 0
        }
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download image from URL
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(
                url, 
                timeout=self.timeout, 
                headers=headers,
                verify=False
            )
            response.raise_for_status()
            
            # Load image from bytes
            img = Image.open(BytesIO(response.content))
            return img
            
        except requests.exceptions.Timeout:
            return None
        except Exception:
            return None
    
    def resize_image_with_padding(
        self,
        img: Image.Image,
        convert_rgb: bool = True
    ) -> Optional[Image.Image]:
        """
        Resize image to target size WITHOUT distortion using padding
        
        Args:
            img: PIL Image object
            convert_rgb: Whether to convert to RGB
            
        Returns:
            Resized PIL Image with padding or None if failed
        """
        try:
            # Convert to RGB if needed
            if convert_rgb and img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling factor to fit image within target size
            original_width, original_height = img.size
            target_width, target_height = self.target_size
            
            # Calculate aspect ratios
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            # Determine new size maintaining aspect ratio
            if original_ratio > target_ratio:
                # Image is wider - fit to width
                new_width = target_width
                new_height = int(target_width / original_ratio)
            else:
                # Image is taller - fit to height
                new_height = target_height
                new_width = int(target_height * original_ratio)
            
            # Resize image maintaining aspect ratio
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and padding color
            new_img = Image.new('RGB', self.target_size, self.padding_color)
            
            # Calculate position to paste resized image (center it)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # Paste resized image onto padded background
            new_img.paste(img_resized, (paste_x, paste_y))
            
            return new_img
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return None
    
    def resize_image_alternative(
        self,
        img: Image.Image,
        convert_rgb: bool = True
    ) -> Optional[Image.Image]:
        """
        Alternative method: Use ImageOps.fit with centering
        This crops the image to fit the target size without distortion
        
        Args:
            img: PIL Image object
            convert_rgb: Whether to convert to RGB
            
        Returns:
            Resized PIL Image or None if failed
        """
        try:
            # Convert to RGB if needed
            if convert_rgb and img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use ImageOps.fit to crop and resize without distortion
            # This will crop the image to fit the aspect ratio, then resize
            img_fitted = ImageOps.fit(
                img, 
                self.target_size, 
                Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)  # Center the crop
            )
            
            return img_fitted
            
        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return None
    
    def process_image(
        self,
        media_url: str,
        post_id: str,
        use_padding: bool = True
    ) -> Optional[Dict]:
        """
        Process a single image
        
        Args:
            media_url: URL to image
            post_id: Post ID for naming
            use_padding: If True, use padding method; if False, use crop method
            
        Returns:
            Dictionary with image info or None if failed
        """
        try:
            # Download image
            img = self._download_image(media_url)
            
            if img is None:
                self.stats["download_failed"] += 1
                return None
            
            # Resize image without distortion
            if use_padding:
                img_resized = self.resize_image_with_padding(img)
            else:
                img_resized = self.resize_image_alternative(img)
                
            if img_resized is None:
                self.stats["processing_failed"] += 1
                return None
            
            # Save processed image
            output_filename = f"{post_id}.jpg"
            output_path = self.images_dir / output_filename
            img_resized.save(output_path, "JPEG", quality=90)
            
            # Create relative path for schema
            relative_path = f"data/02_processed/images/{output_filename}"
            
            self.stats["images_success"] += 1
            
            return {
                "processed_path": relative_path,
                "image_size": list(self.target_size),
                "is_video": False,
                "keyframe_paths": [],
                "processing_method": "padding" if use_padding else "crop"
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
        output_jsonl: str,
        use_padding: bool = True
    ) -> None:
        """
        Process a batch of posts from JSONL file
        
        Args:
            input_jsonl: Path to input JSONL file (raw data)
            output_jsonl: Path to output JSONL file (processed data)
            use_padding: If True, use padding; if False, use crop
        """
        print(f"Input file:  {input_jsonl}")
        print(f"Output file: {output_jsonl}")
        print(f"Max samples: {self.max_samples}")
        print(f"Method:      {'Padding (no distortion)' if use_padding else 'Center crop (no distortion)'}")
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
            
            # Limit samples
            lines = lines[:self.max_samples]
            
            print(f"Total records to process: {len(lines)}")
            print("=" * 60)
            print()
            
            # Process each line
            for line in tqdm(lines, desc="Processing images", ncols=80):
                try:
                    record = json.loads(line.strip())
                    self.stats["total_records"] += 1
                    
                    # Skip if no media
                    if 'media_url' not in record or not record['media_url']:
                        self.stats["skipped_no_media"] += 1
                        continue
                    
                    post_id = record.get('id', f"post_{self.stats['total_records']}")
                    media_url = record['media_url']
                    
                    # Skip videos (no OpenCV available)
                    if self.is_video_url(media_url):
                        self.stats["skipped_video"] += 1
                        continue
                    
                    # Process image
                    image_info = self.process_image(media_url, post_id, use_padding=use_padding)
                    
                    # Add image_info to record if successful
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
            
            # Save processed records
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for record in processed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"✓ Saved to: {output_jsonl}")
            print(f"✓ Records saved: {len(processed_records)}")
            print()
            self._print_statistics()
            
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
    
    print()
    print("=" * 60)
    print("IMAGE PROCESSOR FOR FAKE NEWS DETECTION")
    print("VERSION: No Distortion (Padding Method)")
    print("LIMIT: 200 samples for testing")
    print("=" * 60)
    print()
    
    # Configuration
    INPUT_FILE = "data/01_raw/Fakeddit/dataset_Fakeddit_Processed.jsonl"
    OUTPUT_FILE = "data/02_processed/Fakeddit/dataset_Fakeddit_Processed_200samples.jsonl"
    
    # Choose processing method
    USE_PADDING = True  # True = padding (black bars), False = center crop
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        print()
        print("Please make sure the file exists or update INPUT_FILE path")
        return
    
    file_size = os.path.getsize(INPUT_FILE)
    print(f"✓ Input file found: {INPUT_FILE}")
    print(f"  File size: {file_size:,} bytes")
    print(f"✓ Output will be saved to: {OUTPUT_FILE}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_FILE).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    print()
    
    # Initialize processor
    print("Initializing processor...")
    processor = ImageProcessor(
        target_size=(224, 224),
        output_base_dir="data/02_processed",
        timeout=15,
        max_samples=200,
        padding_color=(0, 0, 0)  # Black padding
    )
    print("✓ Processor initialized")
    print()
    
    # Process batch
    try:
        processor.process_batch(
            input_jsonl=INPUT_FILE,
            output_jsonl=OUTPUT_FILE,
            use_padding=USE_PADDING
        )
        print()
        print("=" * 60)
        print("✓ PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Check the output images to verify no distortion")
        print("2. If you prefer cropping over padding, set USE_PADDING = False")
        print("3. Increase max_samples to process more images")
        print()
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠ INTERRUPTED BY USER")
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