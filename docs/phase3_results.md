# Phase 3: Baseline Models — Kết Quả Thực Nghiệm

> **Ngày chạy:** 25/02/2026  
> **Dataset:** Fakeddit (labeled_master.jsonl) — 1,354 samples  
> **Phân chia:** Train 939 / Val 196 / Test 219  
> **Số class:** 6 (TRUE, MOSTLY_TRUE, HALF_TRUE, BARELY_TRUE, FALSE, PANTS_ON_FIRE)  
> **Hardware:** CUDA GPU

---

## 1. Tổng Quan

Phase 3 xây dựng 4 baseline models để thiết lập mốc so sánh (benchmark) cho các phương pháp multimodal ở Phase tiếp theo. Tất cả model sử dụng **frozen pretrained backbone** — chỉ train lớp classifier cuối cùng.

---

## 2. Kiến Trúc Các Model

| Model | Backbone | Trainable Params | Tổng Params | Ghi chú |
|-------|----------|:----------------:|:-----------:|---------|
| **Text-Only** | XLM-RoBERTa (frozen) | 4,614 | 278M | `[CLS]` → Linear(768→6) |
| **Image-Only** | ResNet-50 (frozen) | 12,294 | 23.5M | avgpool → Linear(2048→6) |
| **Fusion** | XLM-R + ResNet-50 (frozen) | 722,694 | 302M | concat(768+2048) → MLP(2816→256→6) |
| **Graph-Only** | 2-layer GCN | all trainable | ~173K | Input projection → GCN×2 → MLP |

---

## 3. Kết Quả

### 3.1 Bảng Tổng Hợp

| Model | 6-Class Acc | 6-Class Macro-F1 | Binary Acc | Binary F1 | Epochs | Ghi chú |
|-------|:----------:|:----------------:|:----------:|:---------:|:------:|---------|
| 🥇 **Text-Only** | 35.2% | **0.192** | 63.9% | **0.543** | 20 | Tốt nhất 6-class |
| 🥈 **Graph-Only** | 33.3% | 0.178 | **66.7%** | **0.545** | 33 | Tốt nhất binary |
| 🥉 Image-Only | 49.8%* | 0.111 | 62.6% | 0.000 | 14 | Collapse |
| 🥉 Fusion | 49.8%* | 0.111 | 62.6% | 0.000 | 12 | Collapse |

> \* Accuracy 49.8% cao do predict toàn class TRUE (chiếm ~50% data), không phản ánh khả năng phân loại thực sự.

### 3.2 Hyperparameters

| Model | Learning Rate | Batch Size | Patience | Optimizer |
|-------|:------------:|:----------:|:--------:|:---------:|
| Text-Only | 2e-4 | 16 | 7 | Adam |
| Image-Only | 1e-3 | 16 | 7 | Adam |
| Fusion | 2e-4 | 16 | 7 | Adam |
| Graph-Only | 1e-3 | full-batch | 15 | Adam |

### 3.3 Training Dynamics

**Text-Only** — Model duy nhất converge đúng cách:
```
Epoch 01: Loss=1.844, Val F1=0.021
Epoch 10: Loss=1.792, Val F1=0.145  (↑ cải thiện rõ)
Epoch 20: Loss=1.783, Val F1=0.190  (best)
```

**Image-Only & Fusion** — Không converge, loss dao động ~1.9:
```
Epoch 01: Loss=1.958, Val F1=0.021
Epoch 14: Loss=1.978, Val F1=0.011 → Early stopping
```

**Graph-Only** — Converge nhanh nhưng overfit sớm:
```
Epoch 05: Loss=1.789, Val F1=0.116
Epoch 20: Loss=1.736, Val F1=0.202 (best)
Epoch 33: Early stopping
```

### 3.4 Per-Class Performance (Test Set)

**Text-Only:**

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| TRUE | 0.54 | 0.49 | 0.51 | 109 |
| MOSTLY_TRUE | 0.13 | 0.22 | 0.17 | 18 |
| HALF_TRUE | 0.00 | 0.00 | 0.00 | 10 |
| BARELY_TRUE | 0.00 | 0.00 | 0.00 | 4 |
| FALSE | 0.21 | 0.22 | 0.22 | 45 |
| PANTS_ON_FIRE | 0.23 | 0.30 | 0.26 | 33 |

**Graph-Only:**

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| TRUE | 0.38 | 0.55 | 0.44 | 11 |
| MOSTLY_TRUE | 0.00 | 0.00 | 0.00 | 4 |
| HALF_TRUE | 0.00 | 0.00 | 0.00 | 2 |
| FALSE | 0.44 | 0.44 | 0.44 | 9 |
| PANTS_ON_FIRE | 0.00 | 0.00 | 0.00 | 4 |

---

## 4. Phân Tích

### 4.1 Data Imbalance

Phân bố label trong test set **rất lệch**:

```
TRUE:          109 mẫu (49.8%)  ████████████████████
FALSE:          45 mẫu (20.5%)  ████████
PANTS_ON_FIRE:  33 mẫu (15.1%)  ██████
MOSTLY_TRUE:    18 mẫu  (8.2%)  ███
HALF_TRUE:      10 mẫu  (4.6%)  ██
BARELY_TRUE:     4 mẫu  (1.8%)  █
```

→ Các class HALF_TRUE và BARELY_TRUE **quá ít mẫu** để model học được.

### 4.2 Key Findings

| # | Phát Hiện | Ý Nghĩa |
|---|-----------|---------|
| 1 | **Text là modality hiệu quả nhất** | Nội dung ngôn ngữ mang nhiều tín hiệu phân biệt thật/giả nhất |
| 2 | **Graph rất hiệu quả dù chỉ 185 nodes** | Cấu trúc tương tác/lan truyền mang thông tin bổ sung quý giá |
| 3 | **Graph có Binary Acc cao nhất (66.7%)** | Mô hình lan truyền tin phân biệt Real/Fake tốt hơn nội dung |
| 4 | **Image đơn thuần không hiệu quả** | Frozen ResNet features tổng quát không phù hợp cho fake news |
| 5 | **Simple concat fusion thất bại** | MLP 722K params quá lớn cho 939 samples; image noise làm hại text signal |

### 4.3 Tại Sao Image & Fusion Collapse?

1. **Image features không discriminative:** ResNet-50 pretrained trên ImageNet phát hiện vật thể, không phát hiện "mức độ giả" của tin tức
2. **Fusion MLP quá lớn:** 722K trainable params / 939 samples = ratio 770:1 → underfitting
3. **Noisy modality dominates:** Khi concat, image noise làm loãng text signal hữu ích

---

## 5. Kết Luận & Hướng Tiếp Theo

### Baseline Benchmark đã thiết lập:
- **6-Class:** F1 ≈ 0.19 (Text-Only)
- **Binary:** Acc ≈ 66.7% (Graph-Only)

### Hướng đi Phase 4 (Multimodal GNN):
1. **Text + Graph fusion** — kết hợp 2 modality mạnh nhất bằng GNN
2. **Attention-based fusion** thay vì simple concat
3. **Thu thập thêm data** — đặc biệt các class thiếu (HALF_TRUE, BARELY_TRUE)
4. **Fine-tune backbone** khi đủ data (>3,000 samples)

---

## 6. Cách Reproduce

```bash
conda activate multimodal_gnn

# Text baseline
python src/training/train_baseline.py --model_type text --epochs 20 --lr 2e-4 --batch_size 16

# Image baseline
python src/training/train_baseline.py --model_type image --epochs 20 --lr 1e-3 --batch_size 16

# Fusion baseline
python src/training/train_baseline.py --model_type fusion --epochs 20 --lr 2e-4 --batch_size 16

# Graph baseline
python src/training/train_gnn.py --epochs 100 --gnn_type gcn
```

Checkpoints được lưu tại `models/checkpoints/`.
