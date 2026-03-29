"""
PyTorch Dataset cho Fakeddit — hỗ trợ Text, Image, hoặc cả hai.

Đọc từ labeled_master.jsonl, tự phân loại train/val/test theo trường 'split'.
Tương thích với Text-Only, Image-Only, và Fusion baseline.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from src.data.label_utils import BINARY_LABEL_MAP, BINARY_LABEL_NAMES, derive_binary_label

logger = logging.getLogger(__name__)

# ============================================================
# Label Mapping: 6 class (PHASE 3 baseline)
# ============================================================
LABEL_MAP_6 = {
    'TRUE': 0,
    'MOSTLY_TRUE': 1,
    'HALF_TRUE': 2,
    'BARELY_TRUE': 3,
    'FALSE': 4,
    'PANTS_ON_FIRE': 5,
}

# Binary mapping (for binary evaluation):
# 0,1,2 → 0 (Real-ish)  |  3,4,5 → 1 (Fake-ish)
LABEL_NAMES_6 = list(LABEL_MAP_6.keys())
LABEL_NAMES_BINARY = BINARY_LABEL_NAMES

# ============================================================
# Đường dẫn mặc định
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / 'data' / '03_clean' / 'Fakeddit' / 'labeled_master_binary.jsonl'
LEGACY_DATA_PATH = PROJECT_ROOT / 'data' / '03_clean' / 'Fakeddit' / 'labeled_master.jsonl'
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / 'data'


class FakedditTextImageDataset(Dataset):
    """
    Dataset cho Fake News Detection.

    Hỗ trợ 3 chế độ:
      - 'text'  : trả về (input_ids, attention_mask, label)
      - 'image' : trả về (image_tensor, label)
      - 'both'  : trả về (input_ids, attention_mask, image_tensor, label)

    Args:
        data_path: Đường dẫn tới labeled_master.jsonl
        split: 'train', 'val', hoặc 'test'
        mode: 'text', 'image', hoặc 'both'
        tokenizer_name: Tên tokenizer HuggingFace (mặc định xlm-roberta-base)
        max_length: Chiều dài tối đa cho tokenizer
        image_size: Kích thước ảnh đầu ra (mặc định 224)
        image_root: Thư mục gốc chứa ảnh (để chuyển đổi Docker path sang local path)
    """

    def __init__(
        self,
        data_path: str = None,
        split: str = 'train',
        mode: str = 'text',       # 'text', 'image', 'both'
        label_mode: str = 'binary',
        tokenizer_name: str = 'xlm-roberta-base',
        max_length: int = 128,
        image_size: int = 224,
        image_root: str = None,
    ):
        assert split in ('train', 'val', 'test'), f"split phải là 'train', 'val', hoặc 'test', nhận được '{split}'"
        assert mode in ('text', 'image', 'both'), f"mode phải là 'text', 'image', hoặc 'both', nhận được '{mode}'"
        assert label_mode in ('binary', '6class'), f"Invalid label_mode: {label_mode}"

        self.split = split
        self.mode = mode
        self.label_mode = label_mode
        self.max_length = max_length
        self.image_size = image_size

        # Paths
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else LEGACY_DATA_PATH
        self.image_root = Path(image_root) if image_root else DEFAULT_IMAGE_ROOT

        # Load data
        self.records = self._load_records()
        logger.info(f"📂 Loaded {len(self.records)} records for split='{split}' mode='{mode}'")

        # Lazy-load tokenizer và transforms
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name
        self._image_transforms = None

    # ----------------------------------------------------------
    # Đọc JSONL
    # ----------------------------------------------------------
    def _load_records(self) -> List[Dict]:
        """Đọc file JSONL và lọc theo split."""
        records = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Lọc theo split
                if rec.get('split') != self.split:
                    continue

                # Lọc bản ghi có label hợp lệ
                if self.label_mode == 'binary':
                    label_str = rec.get('label_binary', '')
                    if label_str not in BINARY_LABEL_MAP:
                        try:
                            label_str, _ = derive_binary_label(rec)
                            rec['label_binary'] = label_str
                        except ValueError:
                            continue
                    if label_str not in BINARY_LABEL_MAP:
                        continue
                else:
                    label_str = rec.get('label', '')
                    if label_str not in LABEL_MAP_6:
                        continue

                records.append(rec)

        return records

    # ----------------------------------------------------------
    # Tokenizer (lazy load)
    # ----------------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        return self._tokenizer

    # ----------------------------------------------------------
    # Image transforms (lazy load)
    # ----------------------------------------------------------
    @property
    def image_transforms(self):
        if self._image_transforms is None:
            from torchvision import transforms
            self._image_transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        return self._image_transforms

    # ----------------------------------------------------------
    # Chuyển Docker path → local path
    # ----------------------------------------------------------
    def _resolve_image_path(self, record: Dict) -> Optional[Path]:
        """
        Chuyển đổi path kiểu Docker (/data/local-files/?d=02_processed/...)
        sang đường dẫn local thực sự.
        """
        # Thử lấy từ image_info.processed_path
        img_info = record.get('image_info', {})
        raw_path = img_info.get('processed_path', '')

        if not raw_path:
            raw_path = record.get('image', '')

        if not raw_path:
            return None

        # Xử lý Docker-style path: /data/local-files/?d=02_processed/images/...
        if '/data/local-files/?d=' in raw_path:
            # Trích xuất phần sau ?d=
            relative = raw_path.split('?d=')[-1]
        elif raw_path.startswith('/'):
            relative = raw_path.lstrip('/')
        else:
            relative = raw_path

        # Đổi forward slash thành OS separator
        local_path = self.image_root / relative.replace('/', os.sep)
        return local_path

    # ----------------------------------------------------------
    # Load ảnh
    # ----------------------------------------------------------
    def _load_image(self, record: Dict) -> torch.Tensor:
        """Load ảnh và áp dụng transforms. Trả về zero tensor nếu lỗi."""
        from PIL import Image

        img_path = self._resolve_image_path(record)

        if img_path is None or not img_path.exists():
            # Trả về zero tensor
            return torch.zeros(3, self.image_size, self.image_size)

        try:
            img = Image.open(img_path).convert('RGB')
            return self.image_transforms(img)
        except Exception as e:
            logger.warning(f"⚠️ Lỗi load ảnh {img_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

    # ----------------------------------------------------------
    # Tokenize text
    # ----------------------------------------------------------
    def _tokenize_text(self, record: Dict) -> Dict[str, torch.Tensor]:
        """Tokenize clean_text."""
        text = record.get('clean_text', '')
        if not text:
            text = ''

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

    # ----------------------------------------------------------
    # __len__ / __getitem__
    # ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]

        # Label
        if self.label_mode == 'binary':
            label = BINARY_LABEL_MAP[record['label_binary']]
        else:
            label = LABEL_MAP_6[record['label']]
        item = {'label': torch.tensor(label, dtype=torch.long)}

        # Text
        if self.mode in ('text', 'both'):
            text_data = self._tokenize_text(record)
            item['input_ids'] = text_data['input_ids']
            item['attention_mask'] = text_data['attention_mask']

        # Image
        if self.mode in ('image', 'both'):
            item['image'] = self._load_image(record)

        return item


# ==============================================================
# Helper: Tạo DataLoaders
# ==============================================================
def create_dataloaders(
    data_path: str = None,
    mode: str = 'text',
    label_mode: str = 'binary',
    batch_size: int = 16,
    tokenizer_name: str = 'xlm-roberta-base',
    max_length: int = 128,
    image_size: int = 224,
    num_workers: int = 0,     # Windows mặc định 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    loaders = []
    for split in ('train', 'val', 'test'):
        ds = FakedditTextImageDataset(
            data_path=data_path,
            split=split,
            mode=mode,
            label_mode=label_mode,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            image_size=image_size,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        loaders.append(loader)

    return tuple(loaders)


# ==============================================================
# Quick test khi chạy trực tiếp
# ==============================================================
if __name__ == '__main__':
    import sys
    print("=" * 50)
    print("🧪 Testing FakedditTextImageDataset")
    print("=" * 50)

    # Test mode text
    for mode in ['text', 'image', 'both']:
        print(f"\n--- Mode: {mode} ---")
        try:
            ds = FakedditTextImageDataset(split='train', mode=mode)
            print(f"  Train size: {len(ds)}")

            if len(ds) > 0:
                sample = ds[0]
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Test DataLoader
    print("\n--- DataLoader Test (text mode) ---")
    try:
        train_dl, val_dl, test_dl = create_dataloaders(mode='text', batch_size=8)
        print(f"  Train batches: {len(train_dl)}")
        print(f"  Val batches:   {len(val_dl)}")
        print(f"  Test batches:  {len(test_dl)}")

        batch = next(iter(train_dl))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  Batch {k}: shape={v.shape}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print("\n✅ Test hoàn thành!")
