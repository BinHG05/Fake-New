# 📅 KẾ HOẠCH PHASE 3: BASELINE MODELS

> **Mục tiêu:** Xây dựng các mô hình cơ bản (Baseline) để làm chuẩn so sánh cho hệ thống chính.

---

## 1. Tổng quan các Model

Chúng ta sẽ xây dựng 4 mô hình cơ bản, tương ứng với việc sử dụng từng loại dữ liệu riêng lẻ và kết hợp đơn giản.

| Tên Model | Dữ liệu đầu vào | Kiến trúc (Gợi ý) | Mục đích |
|-----------|-----------------|-------------------|----------|
| **Text-Only** | Chỉ văn bản (Title + Text) | XLM-RoBERTa + MLP | Kiểm tra xem thông tin từ văn bản đóng góp bao nhiêu % vào độ chính xác. |
| **Image-Only** | Chỉ hình ảnh (Image) | ResNet50 hoặc CLIP Vision | Kiểm tra xem hình ảnh (fake/real) đóng góp bao nhiêu. |
| **Graph-Only** | Cấu trúc lan truyền (Graph) | GCN hoặc GAT | Kiểm tra xem mô hình có phát hiện được fake news chỉ dựa vào cách nó lan truyền không. |
| **Simple Fusion** | Text + Image | Concatenation (Nối vector) | Mô hình đa phương thức đơn giản nhất để so sánh với các kỹ thuật Fusion phức tạp sau này. |

---

## 2. Quy trình thực hiện (Step-by-Step)

### Tuần 1: Chuẩn bị & Model đơn lẻ

#### Bước 1: Data Loader (Quan trọng nhất)
- [ ] Viết Class `FakedditDataset` trong PyTorch.
- [ ] Input: Đọc từ file `labeled_master.jsonl`.
- [ ] Output của 1 item: `(text_encoding, image_tensor, graph_data, label)`.
- [ ] Chia tập Train/Val/Test cố định (VD: 80% / 10% / 10%).

#### Bước 2: Xây dựng Text-Only Model
- [ ] Load pre-trained `xlm-roberta-base`.
- [ ] Thêm 1 lớp Linear phía sau để phân loại (Real/Fake).
- [ ] Train và lưu kết quả (Accuracy, F1).

#### Bước 3: Xây dựng Image-Only Model
- [ ] Load pre-trained `resnet50`.
- [ ] Train và lưu kết quả.

### Tuần 2: Graph & Fusion

#### Bước 4: Xây dựng Graph-Only Model
- [ ] Dùng thư viện `PyTorch Geometric`.
- [ ] Input: File `.pt` graph đã tạo ở Phase 2.
- [ ] Model: 2 lớp GCNConv -> Pooling -> Linear.

#### Bước 5: Simple Fusion & Đánh giá
- [ ] Ghép vector Text và Image lại.
- [ ] So sánh kết quả của cả 4 model trên tập Test.
- [ ] Vẽ biểu đồ so sánh.

---

## 3. Tiêu chí đánh giá (Metrics)

Với mỗi model, chúng ta cần báo cáo các chỉ số sau:

1.  **Accuracy (Độ chính xác):** Tỷ lệ đoán đúng tổng thể.
2.  **F1-Score (Macro):**  Quan trọng vì dữ liệu có thể bị lệch (imbalanced).
3.  **Confusion Matrix:** Để xem model hay bị nhầm lẫn ở đâu (VD: hay đoán nhầm Fake thành Real).

---

## 4. Output mong đợi của Phase 3

1.  Folder `src/models/baselines/` chứa code sạch của 4 model.
2.  File `reports/phase3_results.md` báo cáo kết quả so sánh.
3.  Checkpoint (file `.pth`) của model tốt nhất để dùng cho demo sau này.
