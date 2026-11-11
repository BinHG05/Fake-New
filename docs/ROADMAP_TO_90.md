# 🏆 Báo cáo Tổng kết & Kế hoạch Phát triển Multimodal GNN (>90%)

## 1. Tổng kết tiến độ hiện tại

Chúng ta đã trải qua 6 phiên bản kiến trúc để tối ưu hóa hiệu suất trên tập dữ liệu 9.5k mẫu (Reddit + Voz + News).

| Giai đoạn | Kiến trúc | Độ chính xác (Test) | Đánh giá |
|---|---|---|---|
| **V1-V3** | Baseline GAT | ~84-85% | Nền tảng ổn định nhưng chưa sâu. |
| **V4-V5** | Global Skip + Transformer | 87.7% - 88.7% | **Tốt nhất.** Giữ được đặc trưng CLIP/XLM-R nguyên bản. |
| **V6 Elite** | Deep Dense GNN (30M params) | ~74% (F1=0) | **Thất bại.** Quá nặng, gây overfit và bão hòa sớm. |
| **Hiện tại** | **V5 Optimized (4.4M params)** | **89.16% (Val)** | **Tiềm năng nhất.** Đang hội tụ rất ổn định với Warmup LR. |

### Các cải tiến quan trọng đã thực hiện:
*   **Global Skip Connection:** Kết nối trực tiếp đặc trưng Multimodal từ đầu vào đến lớp Classifier cuối cùng.
*   **Threshold Tuning:** Tự động tìm ngưỡng quyết định (ví dụ: 0.54 - 0.62) thay vì mặc định 0.5.
*   **Warmup + Cosine Scheduler:** Giúp mô hình ổn định trong 5 epoch đầu trước khi tăng tốc.

---

## 2. Kế hoạch phát triển để vượt mốc 90%

Để vượt qua "bức tường" 89%, chúng ta cần tập trung vào **Chất lượng dữ liệu** và **Sự ổn định** thay vì tăng độ phức tạp của mô hình.

### Chiến lược 1: 5-Fold Cross-Validation Ensemble (Ưu tiên số 1)
*   **Cách làm:** Chia dữ liệu thành 5 phần. Huấn luyện 5 mô hình trên các phần khác nhau. 
*   **Tại sao:** Kết quả hiện tại phụ thuộc khoảng 1-2% vào việc "chia may rủi" tập Train/Val. Ensemble sẽ triệt tiêu sự may rủi này, dự kiến tăng ~1% Accuracy.

### Chiến lược 2: Data Cleaning (Xử lý nhiễu từ Voz)
*   **Cách làm:** Dùng mô hình hiện tại (89%) để dự đoán lại toàn bộ tập Voz. Những mẫu nào mô hình đoán sai với độ tự tin (confidence) > 95% thì khả năng cao là nhãn AI bị sai.
*   **Tại sao:** Loại bỏ nhiễu nhãn là cách duy nhất để mô hình đạt tới sự chính xác tuyệt đối.

### Chiến lược 3: Mixup Augmentation cho Đồ thị
*   **Cách làm:** Tạo ra các đồ thị "ảo" bằng cách trộn đặc trưng của một mẫu REAL và một mẫu FAKE.
*   **Tại sao:** Ép mô hình học được ranh giới quyết định mượt mà hơn, tránh việc ghi nhớ vẹt (memorization) các mẫu nhiễu.

---

## 3. Các bước thực hiện tiếp theo (Next Steps)

1.  **Chạy script `hunt_90_accuracy.py`** với cơ chế lưu **Top-5** mô hình thay vì Top-3.
2.  **Triển khai script `src/utils/clean_voz_labels.py`** để lọc bớt 5% dữ liệu nghi ngờ sai nhãn.
3.  **Tích hợp K-Fold** vào pipeline huấn luyện chính thức.

---
*Tài liệu này được soạn thảo để phục vụ báo cáo NCKH.*
