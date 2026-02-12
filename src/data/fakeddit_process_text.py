"""
Data Processing Pipeline for Fakeddit Dataset (Fake News Detection)
Transforms Fakeddit_pilot_processed_200.jsonl → clean_data (train/val/test splits)

Pipeline Flow:
    01_raw  → 02_processed → 03_clean
    (CORE)     (EXTENDED)    (FINAL SPLITS)

Author: Member C
File: src/data/fakeddit_process_text.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print("=" * 60)
print("FAKEDDIT DATA PROCESSOR - PIPELINE 01 → 02 → 03")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Project root: {project_root}")
print()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("Importing libraries...")

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime

print("  ✓ Basic imports")

try:
    from tqdm import tqdm
    print("  ✓ tqdm")
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    print("  ⚠ tqdm not available, using fallback")

try:
    from sklearn.model_selection import train_test_split
    print("  ✓ sklearn")
except ImportError:
    train_test_split = None
    print("  ⚠ sklearn not available, using simple split")

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError:
    np = None
    print("  ⚠ numpy not available")

print()
print("All imports successful! Starting processor...")
print()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA DEFINITIONS (Local copy for validation)
# =============================================================================

# Fakeddit và LIAR Dataset có các nhãn sau
LABEL_MAPPING = {
    'TRUE': 'True',
    'MOSTLY_TRUE': 'True', 
    'HALF_TRUE': 'True',
    'BARELY_TRUE': 'Fake',
    'FALSE': 'Fake',
    'PANTS_ON_FIRE': 'Fake',
    'PANTS-FIRE': 'Fake',
    # Nếu đã được map sẵn
    'True': 'True',
    'Fake': 'Fake',
    'true': 'True',
    'fake': 'Fake',
}

# CORE_SCHEMA fields (01_raw)
CORE_REQUIRED_FIELDS = ['id', 'timestamp', 'label', 'raw_text', 'media_url', 'user_id']

# EXTENDED_SCHEMA fields (02_processed) 
EXTENDED_REQUIRED_FIELDS = ['id', 'timestamp', 'label', 'clean_text', 'image_info', 'text_features']


class FakedditDataProcessor:
    """
    Process Fakeddit Dataset following the 01 → 02 → 03 pipeline:
    
    01_raw:      Raw JSONL data (CORE_SCHEMA)
    02_processed: Processed data with features (EXTENDED_SCHEMA)  
    03_clean:    Final clean splits (train/val/test)
    """
    
    def __init__(
        self,
        input_file: str,
        output_02_dir: str,
        output_03_dir: str,
        min_text_length: int = 5,
        max_text_length: int = 5000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        Initialize the processor
        
        Args:
            input_file: Path to input JSONL (01_raw)
            output_02_dir: Output directory for 02_processed
            output_03_dir: Output directory for 03_clean
            min_text_length: Minimum clean_text character length
            max_text_length: Maximum clean_text character length
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
        """
        self.input_file = Path(input_file)
        self.output_02_dir = Path(output_02_dir)
        self.output_03_dir = Path(output_03_dir)
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_02_dir.mkdir(parents=True, exist_ok=True)
        self.output_03_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "run_timestamp": self.run_timestamp,
            "step": "01_raw → 02_processed → 03_clean",
            "input_file": str(self.input_file),
            "total_input_records": 0,
            "invalid_json": 0,
            "missing_required_fields": 0,
            "invalid_label": 0,
            "text_too_short": 0,
            "text_too_long": 0,
            "duplicate_ids": 0,
            "valid_02_records": 0,
            "final_03_records": 0
        }
        
        # Track processed IDs for deduplication
        self.processed_ids = set()
        
    def clean_text(self, text: str) -> str:
        """
        Clean raw text - đảm bảo output là lowercase theo EXTENDED_SCHEMA
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text (lowercase, no URLs, normalized whitespace)
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters nhưng giữ dấu câu cơ bản
        text = re.sub(r'[^\w\s.,!?\'-]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase (REQUIRED by EXTENDED_SCHEMA validation)
        text = text.lower()
        
        return text.strip()
    
    def calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment score for text
        Range: -1.0 (negative) to 1.0 (positive)
        
        Note: Placeholder implementation - replace with actual sentiment analysis
        """
        if not text:
            return 0.0
        
        # Simple heuristic based on word patterns
        # TODO: Replace with TextBlob/VADER/transformer-based sentiment
        positive_words = {'good', 'great', 'excellent', 'true', 'correct', 'right', 'honest'}
        negative_words = {'bad', 'false', 'wrong', 'lie', 'fake', 'pants', 'fire'}
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        score = (pos_count - neg_count) / total
        return max(-1.0, min(1.0, score))
    
    def has_caps_lock(self, text: str) -> bool:
        """
        Check if text has CAPS LOCK abuse (more than 50% uppercase letters)
        """
        if not text:
            return False
        
        letters = [c for c in text if c.isalpha()]
        if len(letters) == 0:
            return False
        
        uppercase = sum(1 for c in letters if c.isupper())
        return uppercase > len(letters) * 0.5
    
    def normalize_label(self, label: str) -> Optional[str]:
        """
        Normalize Fakeddit/LIAR labels to Fake/True binary
        
        Args:
            label: Original label from dataset
            
        Returns:
            'Fake' or 'True', or None if invalid
        """
        if not label:
            return None
        
        normalized = LABEL_MAPPING.get(label.upper().replace('-', '_'))
        if not normalized:
            normalized = LABEL_MAPPING.get(label)
        
        return normalized
    
    def generate_id(self, record: Dict) -> str:
        """
        Generate unique ID for record if not present
        """
        if 'id' in record and record['id']:
            return str(record['id'])
        
        # Generate from content hash
        content = f"{record.get('raw_text', '')}{record.get('statement', '')}"
        return f"fakeddit_{hash(content) % 10000000:07d}"
    
    def transform_to_extended(self, record: Dict) -> Optional[Dict]:
        """
        Transform CORE_SCHEMA record to EXTENDED_SCHEMA
        
        Pipeline: 01_raw → 02_processed
        """
        # Generate/validate ID
        record_id = self.generate_id(record)
        
        # Check for duplicates
        if record_id in self.processed_ids:
            self.stats['duplicate_ids'] += 1
            return None
        
        # Get raw text (support multiple field names)
        raw_text = record.get('raw_text') or record.get('statement') or record.get('text') or ''
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        # Validate text length
        if len(clean_text) < self.min_text_length:
            self.stats['text_too_short'] += 1
            return None
        
        if len(clean_text) > self.max_text_length:
            self.stats['text_too_long'] += 1
            return None
        
        # Normalize label
        original_label = record.get('label', '')
        normalized_label = self.normalize_label(original_label)
        
        if not normalized_label:
            self.stats['invalid_label'] += 1
            return None
        
        # Calculate text features
        word_count = len(clean_text.split())
        sentiment_score = self.calculate_sentiment_score(raw_text)
        has_caps = self.has_caps_lock(raw_text)
        
        # Get timestamp (default to current if not present)
        timestamp = record.get('timestamp')
        if not timestamp:
            timestamp = int(time.time())
        
        # Validate timestamp is not in the future
        current_time = int(time.time())
        if timestamp > current_time:
            timestamp = current_time
        
        # Image info logic
        image_info = record.get('image_info')
        if image_info and image_info.get('processed_path'):
            # Ensure image_size exists (Schema requirement)
            if 'image_size' not in image_info and 'width' in image_info and 'height' in image_info:
                image_info['image_size'] = [image_info['width'], image_info['height']]
        else:
            image_info = {
                'processed_path': '',  # Cần chạy fakeddit_preprocessor_image.py để có ảnh
                'image_size': [224, 224],  # Default standard size
                'is_video': False,
                'keyframe_paths': []  # Empty since not video
            }

        # Build EXTENDED_SCHEMA record
        extended_record = {
            # Core identification
            'id': record_id,
            'timestamp': timestamp,
            'label': normalized_label,
            'original_label': original_label,  # Keep original for reference
            
            # Clean text
            'clean_text': clean_text,
            'raw_text': raw_text, # KEEP RAW TEXT FOR LABELING CONTEXT
            
            # Text features (REQUIRED by EXTENDED_SCHEMA)
            'text_features': {
                'word_count': word_count,
                'has_caps_lock': has_caps,
                'sentiment_score': round(sentiment_score, 4)
            },
            
            # Image info
            'image_info': image_info,
            
            # Optional: User features (if available)
            'user_features': {},
            
            # Optional: Graph features (if available)
            'graph_features': {},
            
            # Metadata
            'source_dataset': 'Fakeddit',
            'processed_timestamp': self.run_timestamp
        }
        
        # Add user_id if present
        if record.get('user_id'):
            extended_record['user_id'] = record['user_id']
        
        # Add speaker info if present (optional metadata)
        if record.get('speaker'):
            extended_record['speaker'] = record['speaker']
        if record.get('subject'):
            extended_record['subject'] = record['subject']
        if record.get('context'):
            extended_record['context'] = record['context']
        
        self.processed_ids.add(record_id)
        return extended_record
    
    def split_dataset(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset with stratification by label
        
        Pipeline: 02_processed → 03_clean
        """
        if len(records) == 0:
            return [], [], []
        
        # Separate by label
        fake_records = [r for r in records if r.get('label') == 'Fake']
        true_records = [r for r in records if r.get('label') == 'True']
        
        print(f"  Label distribution: Fake={len(fake_records)}, True={len(true_records)}")
        
        def split_class(class_records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
            if len(class_records) == 0:
                return [], [], []
            
            if len(class_records) < 3:
                # Too few samples, put all in train
                return class_records, [], []
            
            if train_test_split is not None:
                try:
                    # Use sklearn for proper splitting
                    train, temp = train_test_split(
                        class_records,
                        test_size=(1 - self.train_ratio),
                        random_state=42
                    )
                    
                    if len(temp) >= 2:
                        val_ratio_adjusted = self.val_ratio / (1 - self.train_ratio)
                        val, test = train_test_split(
                            temp,
                            test_size=(1 - val_ratio_adjusted),
                            random_state=42
                        )
                    else:
                        val, test = temp, []
                    
                    return train, val, test
                except Exception as e:
                    logger.warning(f"sklearn split failed: {e}, using simple split")
            
            # Fallback: Simple split
            n = len(class_records)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            return class_records[:train_end], class_records[train_end:val_end], class_records[val_end:]
        
        # Split each class
        fake_train, fake_val, fake_test = split_class(fake_records)
        true_train, true_val, true_test = split_class(true_records)
        
        # Combine splits
        train_set = fake_train + true_train
        val_set = fake_val + true_val
        test_set = fake_test + true_test
        
        # Shuffle if numpy available
        if np is not None:
            np.random.seed(42)
            np.random.shuffle(train_set)
            np.random.shuffle(val_set)
            np.random.shuffle(test_set)
        
        # Add split label to each record
        for r in train_set:
            r['split'] = 'train'
        for r in val_set:
            r['split'] = 'val'
        for r in test_set:
            r['split'] = 'test'
        
        return train_set, val_set, test_set
    
    def calculate_statistics(self, records: List[Dict]) -> Dict:
        """Calculate dataset statistics for reporting"""
        if not records:
            return {}
        
        stats = {
            'total_samples': len(records),
            'label_distribution': dict(Counter(r.get('label', 'Unknown') for r in records)),
            'original_label_distribution': dict(Counter(r.get('original_label', 'Unknown') for r in records)),
            'text_stats': {
                'avg_word_count': 0,
                'min_word_count': 0,
                'max_word_count': 0,
                'avg_sentiment': 0
            },
            'split_distribution': dict(Counter(r.get('split', 'Unknown') for r in records))
        }
        
        # Calculate text statistics
        word_counts = [r.get('text_features', {}).get('word_count', 0) for r in records]
        sentiments = [r.get('text_features', {}).get('sentiment_score', 0) for r in records]
        
        if word_counts:
            stats['text_stats']['avg_word_count'] = sum(word_counts) / len(word_counts)
            stats['text_stats']['min_word_count'] = min(word_counts)
            stats['text_stats']['max_word_count'] = max(word_counts)
        
        if sentiments:
            stats['text_stats']['avg_sentiment'] = sum(sentiments) / len(sentiments)
        
        return stats
    
    def process(self):
        """
        Main processing pipeline
        
        01_raw → 02_processed → 03_clean
        """
        
        # =========================================================================
        # STEP 1: Load data from 01_raw
        # =========================================================================
        print("=" * 60)
        print("STEP 1/5: LOADING DATA FROM 01_raw")
        print("=" * 60)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"Reading: {self.input_file}")
        
        raw_records = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                self.stats['total_input_records'] += 1
                
                try:
                    record = json.loads(line.strip())
                    raw_records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    self.stats['invalid_json'] += 1
                    continue
        
        print(f"✓ Loaded {len(raw_records)} records from 01_raw")
        print()
        
        if len(raw_records) == 0:
            raise ValueError("No valid records found in input file")
        
        # =========================================================================
        # STEP 2: Transform to EXTENDED_SCHEMA (01_raw → 02_processed)
        # =========================================================================
        print("=" * 60)
        print("STEP 2/5: TRANSFORMING TO EXTENDED_SCHEMA (→ 02_processed)")
        print("=" * 60)
        
        processed_records = []
        for record in tqdm(raw_records, desc="Processing", ncols=80):
            extended = self.transform_to_extended(record)
            if extended:
                processed_records.append(extended)
        
        self.stats['valid_02_records'] = len(processed_records)
        
        print(f"✓ Transformed {len(processed_records)} records to EXTENDED_SCHEMA")
        print(f"  - Duplicates removed: {self.stats['duplicate_ids']}")
        print(f"  - Invalid labels: {self.stats['invalid_label']}")
        print(f"  - Text too short: {self.stats['text_too_short']}")
        print(f"  - Text too long: {self.stats['text_too_long']}")
        print()
        
        if len(processed_records) == 0:
            raise ValueError("No valid records after processing")
        
        # =========================================================================
        # STEP 3: Save 02_processed output
        # =========================================================================
        print("=" * 60)
        print("STEP 3/5: SAVING 02_processed")
        print("=" * 60)
        
        output_02_file = self.output_02_dir / f"dataset_Fakeddit_{self.run_timestamp}.jsonl"
        
        with open(output_02_file, 'w', encoding='utf-8') as f:
            for record in processed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(processed_records)} records to: {output_02_file}")
        print()
        
        # =========================================================================
        # STEP 4: Split dataset (02_processed → 03_clean)
        # =========================================================================
        print("=" * 60)
        print("STEP 4/5: SPLITTING DATASET (→ 03_clean)")
        print("=" * 60)
        
        train_set, val_set, test_set = self.split_dataset(processed_records)
        
        self.stats['final_03_records'] = len(train_set) + len(val_set) + len(test_set)
        
        print(f"✓ Split complete:")
        total = len(processed_records)
        print(f"  - Train: {len(train_set)} ({len(train_set)/total*100:.1f}%)")
        print(f"  - Val:   {len(val_set)} ({len(val_set)/total*100:.1f}%)")
        print(f"  - Test:  {len(test_set)} ({len(test_set)/total*100:.1f}%)")
        print()
        
        # =========================================================================
        # STEP 5: Save 03_clean outputs
        # =========================================================================
        print("=" * 60)
        print("STEP 5/5: SAVING 03_clean")
        print("=" * 60)
        
        # Create Fakeddit subdirectory
        output_03_fakeddit = self.output_03_dir / "Fakeddit"
        output_03_fakeddit.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in [('train', train_set), ('val', val_set), ('test', test_set)]:
            output_file = output_03_fakeddit / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in split_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"✓ Saved {split_name}.jsonl ({len(split_data)} records)")
        
        # Save statistics
        all_records = train_set + val_set + test_set
        statistics = self.calculate_statistics(all_records)
        statistics['processing_stats'] = self.stats
        
        stats_file = output_03_fakeddit / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved statistics.json")
        
        # Save quality report
        quality_file = output_03_fakeddit / f"quality_report_{self.run_timestamp}.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump({
                'run_timestamp': self.run_timestamp,
                'pipeline': '01_raw → 02_processed → 03_clean',
                'input_file': str(self.input_file),
                'processing_stats': self.stats,
                'final_statistics': statistics
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved quality_report_{self.run_timestamp}.json")
        
        print()
        print("=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        self._print_summary(statistics)
    
    def _print_summary(self, statistics: Dict):
        """Print final summary"""
        print()
        print("SUMMARY:")
        print("-" * 60)
        print(f"Pipeline:               01_raw → 02_processed → 03_clean")
        print(f"Run timestamp:          {self.run_timestamp}")
        print(f"Input file:             {self.input_file.name}")
        print(f"Input records:          {self.stats['total_input_records']}")
        print(f"Valid 02_processed:     {self.stats['valid_02_records']}")
        print(f"Final 03_clean:         {self.stats['final_03_records']}")
        
        if self.stats['total_input_records'] > 0:
            rate = self.stats['final_03_records'] / self.stats['total_input_records'] * 100
            print(f"Success rate:           {rate:.1f}%")
        
        print()
        print("Label distribution:")
        for label, count in statistics.get('label_distribution', {}).items():
            print(f"  - {label}: {count}")
        
        print()
        print("Split distribution:")
        for split, count in statistics.get('split_distribution', {}).items():
            print(f"  - {split.capitalize()}: {count}")
        
        print("-" * 60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Fakeddit text data and create Label Studio files')
    parser.add_argument(
        '--input',
        default=os.path.join("data", "02_processed", "dataset_output.jsonl"),
        help='Path to input JSONL (from image processor)'
    )
    parser.add_argument(
        '--batch-name',
        default=None,
        help='Name of the batch (e.g., "batch_200_400") to create a separate folder in 03_clean'
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("FAKEDDIT DATA PROCESSOR")
    print("Pipeline: 01_raw → 02_processed → 03_clean")
    print("=" * 60)
    print()
    
    # Define paths following the 01/02/03 structure
    INPUT_FILE = args.input
    
    OUTPUT_02_DIR = os.path.join(
        project_root,
        "data", "02_processed", "Fakeddit"
    )
    
    # Nếu có batch_name, tạo folder con trong 03_clean/Fakeddit/
    if args.batch_name:
        OUTPUT_03_DIR = os.path.join(
            project_root,
            "data", "03_clean", "Fakeddit", args.batch_name
        )
    else:
        OUTPUT_03_DIR = os.path.join(
            project_root,
            "data", "03_clean"
        )
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        print()
        print("→ Bạn cần chạy fakeddit_preprocessor_image.py TRƯỚC để tạo file này!")
        print(f"  Command: python src/data/fakeddit_preprocessor_image.py --input <raw_batch_file>")
        print()
        return
    
    # Get file info
    file_size = os.path.getsize(INPUT_FILE)
    record_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for _ in f:
            record_count += 1
    
    print(f"✓ Input file: {INPUT_FILE}")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Total lines: {record_count}")
    print(f"✓ Output 02_processed: {OUTPUT_02_DIR}")
    print(f"✓ Output 03_clean: {OUTPUT_03_DIR}")
    print()
    
    # Initialize processor
    processor = FakedditDataProcessor(
        input_file=INPUT_FILE,
        output_02_dir=OUTPUT_02_DIR,
        output_03_dir=OUTPUT_03_DIR,
        min_text_length=5,
        max_text_length=5000,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Run processing
    try:
        processor.process()
        
        print()
        print("=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print()
        print("1. Review processed data:")
        print(f"   - {OUTPUT_02_DIR}/dataset_Fakeddit_*.jsonl")
        print()
        print("2. Review clean splits:")
        print(f"   - {OUTPUT_03_DIR}/Fakeddit/train.jsonl")
        print(f"   - {OUTPUT_03_DIR}/Fakeddit/val.jsonl")
        print(f"   - {OUTPUT_03_DIR}/Fakeddit/test.jsonl")
        print()
        print("3. Validate with schema:")
        print("   python validate/validate_schema.py")
        print()
        print("4. Track with DVC (optional):")
        print(f"   dvc add {OUTPUT_02_DIR}")
        print(f"   dvc add {OUTPUT_03_DIR}/Fakeddit")
        print()
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠  INTERRUPTED BY USER")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ ERROR: {e}")
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