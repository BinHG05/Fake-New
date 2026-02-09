# 📚 TÀI LIỆU DỰ ÁN (PROJECT DOCUMENTATION)

Chào mừng bạn đến với kho tài liệu của dự án **Multimodal Fake News Detection**.
Dưới đây là danh sách các tài liệu quan trọng được phân loại theo mục đích sử dụng.

---

## 🌟 1. Dành cho Người Mới (Onboarding)

Nếu bạn mới tham gia dự án, hãy bắt đầu từ đây:

*   **[📖 PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)**
    *   **Nội dung:** Giải thích toàn bộ dự án bằng ngôn ngữ đơn giản (không code).
    *   **Dành cho:** Tất cả thành viên mới, người không chuyên kỹ thuật.
    *   **Thành phần chính:** Khái niệm Multimodal, GNN, Quy trình 6 bước, Sản phẩm cuối.

*   **[🏷️ LABELING_GUIDE.md](./LABELING_GUIDE.md)**
    *   **Nội dung:** Hướng dẫn cài đặt Label Studio và cách gán nhãn dữ liệu.
    *   **Dành cho:** Team Data Annotation.
    *   **Thành phần chính:** Command Docker, Cấu hình Cloud Storage, Quy trình làm việc.

---

## 🛠️ 2. Dành cho Developer (Technical)

Nếu bạn cần hiểu sâu về code và cấu trúc hệ thống:

*   **[🏗️ CODEBASE_ANATOMY.md](./CODEBASE_ANATOMY.md)**
    *   **Nội dung:** Bản đồ chi tiết source code. File nào làm nhiệm vụ gì, file nào dùng, file nào bỏ.
    *   **Dành cho:** Dev, Leader.
    *   **Thành phần chính:** List các file Active/Placeholder/Unused.

*   **[🌐 PHASE_2_DEV_GUIDE.md](./PHASE_2_DEV_GUIDE.md)**
    *   **Nội dung:** Hướng dẫn kỹ thuật chi tiết Phase 2 (Crawler & Graph).
    *   **Dành cho:** Team Data, Dev mới.

---

## 📋 3. Quản lý Tiến độ (Management)

Theo dõi trạng thái hiện tại và kế hoạch sắp tới:

*   **[✅ CURRENT_TASK_STATUS.md](./CURRENT_TASK_STATUS.md)**
    *   **Nội dung:** Checklist công việc hiện tại. Biết được team đang làm gì, đã xong gì.
    *   **Cập nhật:** Liên tục theo từng ngày.

*   **[📅 IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)**
    *   **Nội dung:** Kế hoạch kỹ thuật chi tiết cho giai đoạn hiện tại (ví dụ: Batch Automation).
    *   **Dành cho:** Tech Lead planning.

---

## 🚀 Đường dẫn nhanh

### Docker
```powershell
# Chạy Label Studio
docker-compose up -d
```

### Script quan trọng
| Task | Lệnh |
|------|------|
| **Chạy Pipeline Batch 400-600** | `python src/utils/batch_pipeline.py --start 400 --count 200` |
| **Gộp file sau khi gán nhãn** | `python src/utils/convert_ls_export_to_jsonl.py <input> <output> --append` |
| **Train Pilot Model** | `python src/training/train_gnn.py` |

---
> *Tài liệu được cập nhật tự động bởi Antigravity Agent.*
