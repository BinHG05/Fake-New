# 🏗️ Phân tích Cấu trúc Source Code (Codebase Anatomy)

> **Cập nhật:** 2026-02-08
> **Trạng thái:** Phase 1 & Pilot GNN (Hoàn thành), Phase 2-4 (Chờ thực hiện)

Dưới đây là bản đồ chi tiết về toàn bộ mã nguồn của dự án. Tôi đã kiểm tra từng file và phân loại chúng để bạn dễ quản lý.

---

## 🟢 CÁC FILE ĐANG HOẠT ĐỘNG (ACTIVE)
*Các file này đã hoàn thiện chức năng và đang được sử dụng trong quy trình hiện tại.*

### 1. Xử lý Dữ liệu (Phase 1)
| File | Đường dẫn | Chức năng chính |
|------|-----------|-----------------|
| `batch_pipeline.py` | `src/utils/` | **"Nhạc trưởng"**: Chạy tự động toàn bộ quy trình từ raw → clean data. |
| `batch_extractor.py` | `src/utils/` | Cắt nhỏ file dữ liệu lớn thành từng batch (200 mẫu). |
| `fakeddit_preprocessor_image.py` | `src/data/` | Tải ảnh từ URL, resize về 224x224, xử lý ảnh lỗi. |
| `fakeddit_process_text.py` | `src/data/` | Làm sạch text (xóa URL, emoji), chuẩn hóa, tạo file JSONL. |
| `convert_ls_export_to_jsonl.py` | `src/utils/` | Gộp kết quả gán nhãn từ Label Studio vào file tổng (`labeled_master.jsonl`). |
| `merge_splits.py` | `src/utils/` | Chia dữ liệu thành tập Train/Val/Test. |

### 2. Xây dựng Đồ thị & Đặc trưng (Phase 2 - Pilot)
| File | Đường dẫn | Chức năng chính |
|------|-----------|-----------------|
| `embedding_extractor.py` | `src/features/` | Dùng **XLM-R** để tạo vector cho text và **CLIP** cho ảnh. |
| `graph_builder.py` | `src/features/` | Xây dựng đồ thị dựa trên độ tương đồng (Similarity Graph) - *Lưu ý: Đây là đồ thị tạm thời cho Pilot.* |
| `dataloader.py` | `src/data/` | `FakeNewsGraphDataset`: Nạp dữ liệu đồ thị vào mô hình PyTorch. |

### 3. Mô hình & Training (Phase 5 - Pilot)
| File | Đường dẫn | Chức năng chính |
|------|-----------|-----------------|
| `cascade_gnn.py` | `src/models/` | Chứa class `MultiModalFakeNewsGNN`: Kiến trúc GNN chính (hỗ trợ GAT, SAGE, GCN). |
| `train_gnn.py` | `src/training/` | Script training chính hiện tại (hỗ trợ Early Stopping, tính Class Weights). |

### 4. Tiện ích (Utils)
| File | Đường dẫn | Chức năng chính |
|------|-----------|-----------------|
| `logger.py` | `src/utils/` | Cấu hình log in ra màn hình/file. |
| `debug_ls_json.py` | `src/utils/` | Kiểm tra file JSON Label Studio nếu bị lỗi. |

---

## 🟡 CÁC FILE CHỜ KÍCH HOẠT (PLACEHOLDERS) 🚧
*Các file này ĐÃ TẠO nhưng CHƯA CÓ CODE (hoặc chỉ có khung `pass`). Sẽ dùng trong các Phase tiếp theo.*

### Phase 2B: Cascade Graph (Sắp làm)
*   **`src/data/reddit_crawler.py`**: Hiện tại chỉ crawl bài lẻ. Cần nâng cấp để lấy comment tree.
*   **`src/features/cascade_graph_builder.py`**: Sẽ dùng để xây dựng đồ thị lan truyền thật từ dữ liệu crawler.

### Phase 3: Baseline Models (Sắp làm)
*   **`src/models/baseline_text.py`**: Chứa model BERT thuần (để so sánh).
*   **`src/models/baseline_image.py`**: Chứa model ViT/ResNet thuần.
*   **`src/training/train_text_baseline.py`**: Script train riêng cho text.
*   **`src/training/train_image_baseline.py`**: Script train riêng cho ảnh.

### Phase 4: Multimodal & Propagation (Tương lai)
*   **`src/models/advanced_gnn.py`**: Cho các model phức tạp hơn (HGT, GraphSAGE biến thể).
*   **`src/models/multimodal_fusion.py`**: Module kết hợp Text + Ảnh chuyên sâu (Cross-Attention).
*   **`src/features/propagation_extractor.py`**: Trích xuất đặc trưng lan truyền (tốc độ share, độ sâu cây).

---

## 🔴 CÁC FILE DƯ THỪA / CẦN XÓA (UNUSED) 🗑️
*Có thể xóa để gọn project.*

| File | Lý do |
|------|-------|
| `src/data/liar_mapper.py` | Dành cho bộ dữ liệu LIAR, chúng ta đang dùng Fakeddit. |
| `src/utils/file_utils.py` | Quá nhỏ (16 bytes), có thể gộp vào `logger.py` hoặc xóa. |
| `src/utils/model_utils.py` | Quá nhỏ, chưa dùng. |

---

## 💡 Tổng kết & Lời khuyên

1.  **Project hiện tại RẤT GỌN:** Chỉ có nhóm 🟢 là đang chạy thật.
2.  **Nhóm 🟡 là bản đồ tương lai:** Chúng ta đã đặt chỗ sẵn cho các phần việc sắp tới (Baseline, Cascade). Không cần xóa, cứ để đó để nhớ việc cần làm.
3.  **Hành động tiếp theo:**
    *   Tập trung code vào nhóm 🟡 **Phase 3 (Baseline)** trước.
    *   Sau đó đến nhóm 🟡 **Phase 2B (Cascade Crawler)**.
