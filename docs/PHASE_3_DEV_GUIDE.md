# 🧑‍💻 HƯỚNG DẪN KỸ THUẬT PHASE 3 (DEV GUIDE)

Tài liệu này hướng dẫn chi tiết cho các bạn Developer thực hiện code các model trong Phase 3.

---

## 📂 Cấu trúc thư mục mới

Chúng ta sẽ tạo folder mới là `src/models/baselines`. Cấu trúc file sẽ như sau:

```
src/
├── data/
│   ├── dataset.py          # (Cần tạo) Class Dataset chính
│   └── ...
├── models/
│   └── baselines/          # (Mới)
│       ├── __init__.py
│       ├── text_only.py    # Class TextOnlyModel
│       ├── image_only.py   # Class ImageOnlyModel
│       ├── graph_only.py   # Class GraphOnlyModel
│       └── fusion.py       # Class SimpleFusionModel
└── main_baseline.py        # (Mới) Script chạy training
```

---

## 🛠️ Hướng dẫn code từng phần

### 1. File `src/data/dataset.py`

Đây là **xương sống** của việc training. Bạn cần viết class kế thừa từ `torch.utils.data.Dataset`.

**Yêu cầu:**
- Hàm `__init__`: Nhận list các bản ghi (từ file jsonl).
- Hàm `__getitem__(index)`:
    - Lấy text -> Tokenize bằng Tokenizer -> Trả về `input_ids`, `attention_mask`.
    - Lấy đường dẫn ảnh -> Load ảnh -> Resize (224x224) -> Trả về Tensor.
    - Lấy label -> Trả về Tensor (0 hoặc 1).

### 2. Các file Model (`src/models/baselines/*.py`)

Mỗi file chỉ chứa 1 Class đơn giản kế thừa từ `nn.Module`.

**Ví dụ `text_only.py`:**
```python
class TextOnlyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained('xlm-roberta-base')
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy vector [CLS] token đầu tiên
        cls_token = out.last_hidden_state[:, 0, :]
        return self.fc(cls_token)
```

### 3. Training Script (`main_baseline.py`)

File này sẽ điều phối việc chạy code.

- **Agruments cần có:**
    - `--model_type`: Chọn loại model để train ('text', 'image', 'graph', 'fusion').
    - `--epochs`: Số vòng lặp (VD: 10, 20).
    - `--batch_size`: Kích thước batch (VD: 16, 32).
    - `--lr`: Learning rate (thường là 2e-5 với BERT).

- **Luồng chạy:**
    1.  Load Config & Arguments.
    2.  Chuẩn bị Dataset & DataLoader.
    3.  Khởi tạo Model (dựa vào `model_type` được chọn).
    4.  Chạy vòng lặp Train (Forward -> Loss -> Backward -> Optimizer).
    5.  Chạy Evaluate sau mỗi epoch.
    6.  Lưu model nếu kết quả tốt nhất.

---

## ✅ Checklist kiểm tra (Definition of Done)

Trước khi báo cáo hoàn thành, hãy tự kiểm tra:

1.  [ ] Code chạy được không lỗi cú pháp.
2.  [ ] Train thử 1 epoch thấy Loss giảm.
3.  [ ] Code có comment giải thích các đoạn phức tạp.
4.  [ ] File output model (`.pth`) được lưu đúng chỗ.
