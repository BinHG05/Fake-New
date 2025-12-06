"""
Data Cleaning Pipeline for Fake News Detection
Transforms processed_data → clean_data
Author: Member C
File: src/data/data_cleaner.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print("=" * 60)
print("DATA CLEANER - QUALITY ASSURANCE VERSION")
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
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import Counter

print("  ✓ Basic imports")

from PIL import Image
import imagehash
print("  ✓ PIL & imagehash")

from tqdm import tqdm
print("  ✓ tqdm")

from sklearn.model_selection import train_test_split
print("  ✓ sklearn")

import numpy as np
print("  ✓ numpy")

print()
print("All imports successful! Starting cleaner...")
print()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and prepare data for training"""
    
    def __init__(
        self,
        input_jsonl: str,
        output_dir: str,
        min_text_length: int = 5,
        max_text_length: int = 500,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """Initialize the cleaner"""
        self.input_jsonl = Path(input_jsonl)
        self.output_dir = Path(output_dir)
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        self.CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
        
        self.stats = {
            "total_input_records": 0,
            "corrupted_images": 0,
            "missing_images": 0,
            "wrong_size": 0,
            "duplicates_found": 0,
            "text_too_short": 0,
            "text_too_long": 0,
            "final_valid_records": 0
        }
        
        self.issues = {
            'corrupted': [],
            'missing': [],
            'wrong_size': [],
            'duplicates': []
        }
    
    def clean_text(self, text: str) -> str:
        """Clean raw text field"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def verify_image(self, img_path: str, record_id: str) -> Tuple[bool, str]:
        """Verify image quality"""
        # Convert to absolute path if needed
        if not os.path.isabs(img_path):
            img_path = os.path.join(project_root, img_path)
        
        if not os.path.exists(img_path):
            self.issues['missing'].append(record_id)
            return False, "missing"
        
        try:
            img = Image.open(img_path)
            img.load()  # Force load to detect corrupted images
            
            if img.size != (224, 224):
                self.issues['wrong_size'].append(record_id)
                return False, "wrong_size"
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.close()
            return True, ""
            
        except Exception as e:
            logger.warning(f"Error verifying image {record_id}: {e}")
            self.issues['corrupted'].append(record_id)
            return False, "corrupted"
    
    def find_duplicates(self, records: List[Dict]) -> List[Dict]:
        """Find and remove duplicate images"""
        print("  Calculating image hashes...")
        
        hashes = {}
        unique_records = []
        duplicate_pairs = []
        
        for record in tqdm(records, desc="  Checking duplicates", ncols=80):
            img_path = record['image_info']['processed_path']
            record_id = record['id']
            
            # Convert to absolute path if needed
            if not os.path.isabs(img_path):
                img_path = os.path.join(project_root, img_path)
            
            try:
                img = Image.open(img_path)
                img_hash = str(imagehash.phash(img))
                img.close()
                
                if img_hash in hashes:
                    duplicate_pairs.append({
                        'original': hashes[img_hash],
                        'duplicate': record_id
                    })
                    self.stats['duplicates_found'] += 1
                else:
                    hashes[img_hash] = record_id
                    unique_records.append(record)
                    
            except Exception as e:
                logger.warning(f"Error hashing {record_id}: {e}")
                continue
        
        self.issues['duplicates'] = duplicate_pairs
        
        return unique_records
    
    def add_metadata(self, record: Dict) -> Dict:
        """Add additional metadata"""
        # Clean text
        cleaned_text = self.clean_text(record.get('raw_text', ''))
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        
        record['cleaned_text'] = cleaned_text
        record['word_count'] = word_count
        
        # Add CLIP normalization info
        record['clip_norm'] = {
            'mean': self.CLIP_MEAN,
            'std': self.CLIP_STD,
            'input_size': 224
        }
        
        return record
    
    def split_dataset(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset with stratification"""
        # Separate by label
        fake_records = [r for r in records if r.get('label') == 'Fake']
        true_records = [r for r in records if r.get('label') == 'True']
        
        print(f"  Fake samples: {len(fake_records)}")
        print(f"  True samples: {len(true_records)}")
        
        def split_class(class_records):
            if len(class_records) == 0:
                return [], [], []
            
            # Split train and temp (val+test)
            train, temp = train_test_split(
                class_records, 
                test_size=(1 - self.train_ratio),
                random_state=42
            )
            
            # Split temp into val and test
            if len(temp) > 1:
                val_ratio_adjusted = self.val_ratio / (1 - self.train_ratio)
                val, test = train_test_split(
                    temp,
                    test_size=(1 - val_ratio_adjusted),
                    random_state=42
                )
            else:
                val = temp
                test = []
            
            return train, val, test
        
        # Split each class
        fake_train, fake_val, fake_test = split_class(fake_records)
        true_train, true_val, true_test = split_class(true_records)
        
        # Combine splits
        train_set = fake_train + true_train
        val_set = fake_val + true_val
        test_set = fake_test + true_test
        
        # Shuffle each split
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
        """Calculate dataset statistics"""
        stats = {
            'total_samples': len(records),
            'label_distribution': Counter(r.get('label', 'Unknown') for r in records),
            'text_stats': {
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'length_distribution': {}
            },
            'image_stats': {
                'total_size_mb': 0,
                'avg_file_size_kb': 0
            },
            'split_distribution': {}
        }
        
        # Calculate text statistics
        lengths = [r.get('word_count', 0) for r in records]
        if lengths:
            stats['text_stats']['avg_length'] = sum(lengths) / len(lengths)
            stats['text_stats']['min_length'] = min(lengths)
            stats['text_stats']['max_length'] = max(lengths)
            
            # Create length distribution bins
            length_bins = [0, 10, 20, 50, 100, 200, 500]
            for i in range(len(length_bins) - 1):
                bin_name = f"{length_bins[i]}-{length_bins[i+1]}"
                count = sum(1 for l in lengths 
                           if length_bins[i] <= l < length_bins[i+1])
                stats['text_stats']['length_distribution'][bin_name] = count
            
            # Add 500+ bin
            count_500_plus = sum(1 for l in lengths if l >= 500)
            stats['text_stats']['length_distribution']['500+'] = count_500_plus
        
        # Calculate image statistics
        total_size = 0
        valid_files = 0
        for record in records:
            img_path = record['image_info']['processed_path']
            if not os.path.isabs(img_path):
                img_path = os.path.join(project_root, img_path)
            
            if os.path.exists(img_path):
                total_size += os.path.getsize(img_path)
                valid_files += 1
        
        if valid_files > 0:
            stats['image_stats']['total_size_mb'] = total_size / (1024 * 1024)
            stats['image_stats']['avg_file_size_kb'] = (total_size / valid_files) / 1024
        
        # Calculate split distribution
        if records and 'split' in records[0]:
            stats['split_distribution'] = dict(Counter(r.get('split', 'Unknown') for r in records))
        
        return stats
    
    def process(self):
        """Main processing pipeline"""
        
        print("=" * 60)
        print("STEP 1/7: LOADING DATA")
        print("=" * 60)
        
        if not self.input_jsonl.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_jsonl}")
        
        records = []
        with open(self.input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # Validate required fields
                    if 'id' in record and 'image_info' in record and 'raw_text' in record:
                        records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line: {e}")
                    continue
        
        self.stats['total_input_records'] = len(records)
        print(f"✓ Loaded {len(records)} records")
        print()
        
        if len(records) == 0:
            raise ValueError("No valid records found in input file")
        
        # STEP 2: VERIFY IMAGE QUALITY
        print("=" * 60)
        print("STEP 2/7: VERIFYING IMAGE QUALITY")
        print("=" * 60)
        
        valid_records = []
        for record in tqdm(records, desc="Verifying images", ncols=80):
            img_path = record['image_info']['processed_path']
            record_id = record['id']
            
            is_valid, error = self.verify_image(img_path, record_id)
            
            if is_valid:
                valid_records.append(record)
            else:
                if error == "corrupted":
                    self.stats['corrupted_images'] += 1
                elif error == "missing":
                    self.stats['missing_images'] += 1
                elif error == "wrong_size":
                    self.stats['wrong_size'] += 1
        
        print(f"✓ Valid images: {len(valid_records)}")
        print(f"  - Corrupted: {self.stats['corrupted_images']}")
        print(f"  - Missing: {self.stats['missing_images']}")
        print(f"  - Wrong size: {self.stats['wrong_size']}")
        print()
        
        if len(valid_records) == 0:
            raise ValueError("No valid images found after verification")
        
        records = valid_records
        
        # STEP 3: REMOVE DUPLICATES
        print("=" * 60)
        print("STEP 3/7: REMOVING DUPLICATES")
        print("=" * 60)
        
        records = self.find_duplicates(records)
        print(f"✓ Unique images: {len(records)}")
        print(f"  - Duplicates removed: {self.stats['duplicates_found']}")
        print()
        
        # STEP 4: CLEAN TEXT
        print("=" * 60)
        print("STEP 4/7: CLEANING TEXT")
        print("=" * 60)
        
        cleaned_records = []
        for record in tqdm(records, desc="Cleaning text", ncols=80):
            record = self.add_metadata(record)
            
            word_count = record['word_count']
            if word_count < self.min_text_length:
                self.stats['text_too_short'] += 1
                continue
            elif word_count > self.max_text_length:
                self.stats['text_too_long'] += 1
                continue
            
            cleaned_records.append(record)
        
        print(f"✓ Valid text: {len(cleaned_records)}")
        print(f"  - Too short: {self.stats['text_too_short']}")
        print(f"  - Too long: {self.stats['text_too_long']}")
        print()
        
        if len(cleaned_records) == 0:
            raise ValueError("No valid records after text cleaning")
        
        records = cleaned_records
        self.stats['final_valid_records'] = len(records)
        
        # STEP 5: SPLIT DATASET
        print("=" * 60)
        print("STEP 5/7: SPLITTING DATASET")
        print("=" * 60)
        
        train_set, val_set, test_set = self.split_dataset(records)
        
        print(f"✓ Split complete:")
        print(f"  - Train: {len(train_set)} ({len(train_set)/len(records)*100:.1f}%)")
        print(f"  - Val:   {len(val_set)} ({len(val_set)/len(records)*100:.1f}%)")
        print(f"  - Test:  {len(test_set)} ({len(test_set)/len(records)*100:.1f}%)")
        print()
        
        # STEP 6: CALCULATE STATISTICS
        print("=" * 60)
        print("STEP 6/7: CALCULATING STATISTICS")
        print("=" * 60)
        
        all_records = train_set + val_set + test_set
        statistics = self.calculate_statistics(all_records)
        
        print(f"✓ Statistics calculated")
        print(f"  - Total samples: {statistics['total_samples']}")
        print(f"  - Fake: {statistics['label_distribution'].get('Fake', 0)}")
        print(f"  - True: {statistics['label_distribution'].get('True', 0)}")
        print(f"  - Avg text length: {statistics['text_stats']['avg_length']:.1f} words")
        print()
        
        # STEP 7: SAVE CLEAN DATA
        print("=" * 60)
        print("STEP 7/7: SAVING CLEAN DATA")
        print("=" * 60)
        
        for split_name, split_data in [
            ('train', train_set),
            ('val', val_set),
            ('test', test_set)
        ]:
            output_file = self.output_dir / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in split_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"✓ Saved {split_name}.jsonl ({len(split_data)} records)")
        
        # Save statistics
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved statistics.json")
        
        # Save quality report
        quality_file = self.output_dir / "quality_report.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_stats': self.stats,
                'issues': self.issues
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved quality_report.json")
        
        print()
        print("=" * 60)
        print("CLEANING COMPLETE!")
        print("=" * 60)
        print()
        
        # Print final summary
        self._print_summary(statistics)
    
    def _print_summary(self, statistics: Dict):
        """Print final summary"""
        print("SUMMARY:")
        print("-" * 60)
        print(f"Input records:          {self.stats['total_input_records']}")
        print(f"Final valid records:    {self.stats['final_valid_records']}")
        print(f"Success rate:           {self.stats['final_valid_records']/self.stats['total_input_records']*100:.1f}%")
        print()
        print("Data splits:")
        for split_name, count in statistics['split_distribution'].items():
            print(f"  - {split_name.capitalize()}: {count}")
        print()
        print("Label distribution:")
        for label, count in statistics['label_distribution'].items():
            print(f"  - {label}: {count}")
        print("-" * 60)


def main():
    """Main function"""
    
    print()
    print("=" * 60)
    print("DATA CLEANER FOR FAKE NEWS DETECTION")
    print("=" * 60)
    print()
    
    # Define paths
    INPUT_FILE = os.path.join(
        project_root, 
        "data", "02_processed", "Fakeddit", 
        "dataset_Fakeddit_Processed_200samples.jsonl"
    )
    OUTPUT_DIR = os.path.join(project_root, "data","02_processed", "clean", "Fakeddit")
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        print()
        print("Please run preprocessor_image.py first!")
        print()
        print("Expected file structure:")
        print("  data/02_processed/Fakeddit/dataset_Fakeddit_Processed_200samples.jsonl")
        return
    
    print(f"✓ Input file: {INPUT_FILE}")
    print(f"✓ Output dir: {OUTPUT_DIR}")
    print()
    
    # Initialize cleaner
    cleaner = DataCleaner(
        input_jsonl=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        min_text_length=5,
        max_text_length=500,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Run processing
    try:
        cleaner.process()
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