# 📅 KẾ HOẠCH PHASE 3: BASELINE MODELS

> **Mục tiêu:** Xây dựng 4 mô hình cơ bản (Baseline) để làm chuẩn so sánh.
> **Dữ liệu hiện tại:** ~400 samples (đủ để test pipeline, chưa đủ cho kết quả chính thức).

---

## 📊 Tổng quan 4 Models cần xây

| # | Model | Input | Output | File code |
|---|-------|-------|--------|-----------|
| 1 | **Text-Only** | Văn bản (title + text) | Real/Fake | `src/models/baselines/text_only.py` |
| 2 | **Image-Only** | Hình ảnh | Real/Fake | `src/models/baselines/image_only.py` |
| 3 | **Graph-Only** | Đồ thị lan truyền (.pt) | Real/Fake | `src/models/baselines/graph_only.py` |
| 4 | **Simple Fusion** | Text + Image (nối vector) | Real/Fake | `src/models/baselines/fusion.py` |

---

## 🔧 PHÂN CHIA THÀNH 5 PHẦN

### PHẦN 1: Data Loader cho Text & Image ⚡ (Làm đầu tiên)
> **Tại sao quan trọng?** Tất cả các model đều phụ thuộc vào phần này.

**Hiện trạng:** Chỉ có `src/data/dataloader.py` cho Graph data. Chưa có Data Loader cho Text/Image.

**Việc cần làm:**
- [ ] Tạo file `src/data/text_image_dataset.py`
    - Class `FakedditTextImageDataset(torch.utils.data.Dataset)`
    - Đọc từ `data/03_clean/Fakeddit/labeled_master.jsonl`
    - `__getitem__()` trả về: `(input_ids, attention_mask, image_tensor, label)`
    - Chia Train/Val/Test (80/10/10) với random seed cố định
- [ ] Test thử: Load được 1 batch không lỗi

**Thư viện cần:**
```
pip install transformers torchvision Pillow
```

---

### PHẦN 2: Text-Only Baseline 📝
> **Độ khó:** ⭐⭐ (Trung bình)

**Việc cần làm:**
- [ ] Hoàn thiện `src/models/baselines/text_only.py`
    - Load `xlm-roberta-base` (frozen hoặc finetune)
    - Thêm 1 lớp Linear(768 → num_classes) phía sau
- [ ] Tạo `src/training/train_text_baseline.py`
    - Training loop: Forward → Loss → Backward → Optimizer
    - Evaluate sau mỗi epoch
    - Lưu model tốt nhất
- [ ] Chạy thử: Train 5-10 epochs, ghi nhận Accuracy & F1

**Lệnh chạy dự kiến:**
```bash
python src/training/train_text_baseline.py --epochs 10 --lr 2e-5 --batch_size 16
```

---

### PHẦN 3: Image-Only Baseline 🖼️
> **Độ khó:** ⭐⭐ (Trung bình)

**Việc cần làm:**
- [ ] Hoàn thiện `src/models/baselines/image_only.py`
    - Load `resnet50` pretrained (ImageNet)
    - Thay lớp cuối: Linear(2048 → num_classes)
- [ ] Tạo `src/training/train_image_baseline.py`
    - Tương tự Text nhưng input là ảnh (224×224)
- [ ] Chạy thử: Train 5-10 epochs

**Lưu ý:** Cần kiểm tra xem ảnh trong `data/02_processed/images/` đã resize chưa.

---

### PHẦN 4: Graph-Only Baseline 🔗
> **Độ khó:** ⭐⭐⭐ (Khó hơn một chút)

**Hiện trạng:** Đã có `src/training/train_gnn.py` hoạt động được! Đây chính là Graph Baseline.

**Việc cần làm:**
- [ ] Hoàn thiện `src/models/baselines/graph_only.py`
    - GCN 2 lớp đơn giản (tách riêng khỏi `cascade_gnn.py`)
- [ ] Tái sử dụng logic từ `train_gnn.py` đã có
- [ ] Chạy lại trên dữ liệu mới nhất và ghi nhận kết quả

**Lưu ý:** Phần này gần như đã xong nhờ code `train_gnn.py` từ Phase trước.

---

### PHẦN 5: Simple Fusion + So sánh tổng hợp 🔀
> **Độ khó:** ⭐⭐⭐ (Cần hoàn thành Phần 2 & 3 trước)

**Việc cần làm:**
- [ ] Hoàn thiện `src/models/baselines/fusion.py`
    - Lấy vector text (768d) + vector image (2048d) → Nối lại → MLP → Classify
- [ ] Tạo script training cho Fusion
- [ ] **So sánh tổng hợp:**
    - [ ] Tổng hợp kết quả 4 models vào bảng
    - [ ] Vẽ Confusion Matrix cho từng model
    - [ ] Viết báo cáo `reports/phase3_results.md`

---

## 📋 THỨ TỰ THỰC HIỆN (Roadmap)

```
Phần 1 (Data Loader)     ← LÀM ĐẦU TIÊN (tất cả phụ thuộc vào đây)
    ↓
Phần 2 (Text) ──┐
Phần 3 (Image) ─┤        ← CÓ THỂ LÀM SONG SONG
Phần 4 (Graph) ─┘
    ↓
Phần 5 (Fusion + Báo cáo) ← LÀM CUỐI CÙNG
```

---

## 📏 Tiêu chí đánh giá (Metrics)

| Chỉ số | Ý nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ đoán đúng tổng thể |
| **F1-Score (Macro)** | Trung bình F1 của tất cả các class (quan trọng khi data lệch) |
| **Confusion Matrix** | Ma trận nhầm lẫn: model hay nhầm ở đâu? |

---

## ⚠️ Lưu ý quan trọng

1. **Random Seed = 42** → Cố định để kết quả lặp lại được.
2. **400 samples là ít** → Kết quả chỉ để sanity check, không đưa vào báo cáo chính thức.
3. **GPU không bắt buộc** → 400 samples train trên CPU cũng nhanh.
4. **Freeze backbone khi ít data** → Với 400 samples, nên freeze BERT/ResNet, chỉ train lớp Linear cuối.
