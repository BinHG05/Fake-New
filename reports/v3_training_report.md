# 🏆 Báo cáo kết quả huấn luyện Multimodal GNN V3
**Ngày thực hiện:** 2026-05-04
**Tổng số dữ liệu:** 9,584 bản ghi (Reddit, Voz, News)
**Cấu hình:** Epochs: 100, Batch Size: 32

## 1. Kết quả chi tiết các mô hình

| Model | Binary Accuracy | Binary F1-Score | Ghi chú |
| :--- | :---: | :---: | :--- |
| **Text Baseline** | 0.7420 | 0.6624 | Sử dụng XLM-RoBERTa features |
| **Image Baseline** | 0.2875 | 0.4466 | Sử dụng CLIP Image features |
| **Fusion Baseline** | 0.7543 | 0.6429 | Kết hợp Early Fusion (Ảnh + Chữ) |
| **Graph GNN** | 0.7420 | 0.6769 | Sử dụng cấu trúc Cascade (GAT) |
| **Multimodal GNN V3** | **0.8649** | **0.7773** | **Mô hình đề xuất (V3)** |

## 2. Biểu đồ so sánh (Tóm tắt)
- **Độ chính xác cao nhất:** 86.49% (Multimodal GNN V3)
- **Mức tăng trưởng (Delta):** 
    - So với Text-only: **+12.29%**
    - So với Graph-only: **+12.29%**
- **F1-Score cải thiện:** **+10.04%** so với Graph GNN.

## 3. Đánh giá chuyên môn
Mô hình **Multimodal GNN V3** thể hiện sự vượt trội hoàn toàn. Việc sử dụng `cross_attention` để kết hợp đặc trưng hình ảnh và văn bản vào các node trong đồ thị lan truyền (Cascade) đã giúp mô hình hiểu được ngữ cảnh phức tạp của tin giả trên các diễn đàn như Voz và Reddit.

**Kết luận:** Hệ thống đã sẵn sàng để triển khai thực tế hoặc viết bài công bố khoa học.
