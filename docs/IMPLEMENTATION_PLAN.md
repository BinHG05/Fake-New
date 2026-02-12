# 🚀 Phase 2 Execution Plan: Data Enrichment & Propagation Modeling

Tài liệu này hướng dẫn chi tiết các bước **chạy code** để thực hiện Phase 2: Từ việc lấy dữ liệu lan truyền (Cascade) đến việc đóng gói thành các đồ thị (Graph) cho mô hình AI.

Dựa trên phân tích các file source code:
1.  `src/data/crawler_enrich.py`
2.  `build_final_graphs.py`

---

## 📋 Tổng quan Quy trình (Workflow)

| Bước | Tên nhiệm vụ | Script thực hiện | Input (Đầu vào) | Output (Đầu ra) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Làm giàu dữ liệu (Enrichment)** | `src/data/crawler_enrich.py` | `data/03_clean/Fakeddit/labeled_master.jsonl` | `data/reddit_enriched_data.jsonl` |
| **2** | **Xây dựng đồ thị (Graph Building)** | `build_final_graphs.py` | `data/reddit_enriched_data.jsonl` | `data/processed_graphs/*.pt` |

---

## 🛠️ Chi tiết từng bước

### 1. Bước 1: Thu thập dữ liệu lan truyền (Enrichment)
**Mục tiêu:** Lấy toàn bộ cây bình luận (Comment Tree) của các bài viết đã được gán nhãn. Đây chính là cấu trúc "lan truyền" của tin tức.

*   **Logic:** Script sẽ đọc file dữ liệu gốc (`labeled_master.jsonl`), lấy ID bài viết, sau đó dùng `RedditCrawler` để tải toàn bộ comment về.
*   **Lệnh thực thi:**
    ```bash
    python src/data/crawler_enrich.py
    ```
*   **Lưu ý:**
    *   Quá trình này có thể tốn thời gian do API Rate Limit của Reddit.
    *   Script có cơ chế **Resume**: Nếu chạy lại, nó sẽ tự động bỏ qua các bài đã tải xong.

### 2. Bước 2: Tạo mô hình lan truyền (Propagation Graphs)
**Mục tiêu:** Chuyển đổi dữ liệu thô (Text, Image, Comment structure) thành dạng đồ thị `.pt`.

*   **Logic:**
    *   Sử dụng `embedding_extractor` để chuyển đổi nội dung bài viết và comment thành vector số.
    *   Xây dựng file đồ thị `.pt` cho mỗi bài viết (Cascade Graph).
*   **Lệnh thực thi:**
    ```bash
    python build_final_graphs.py
    ```
*   **Kết quả:**
    *   Dữ liệu đồ thị từng bài viết sẽ nằm trong `data/processed_graphs/`.
    *   Lưu ý: File đồ thị tổng hợp (`graph.pt`) nếu đã có ở `data/04_graph/` thì đó là Interaction Graph (Phase 2A), khác với Cascade Graph (Phase 2B) này.

---

## ⏭️ Bước tiếp theo: Phase 3 (Baseline Training)

Sau khi hoàn thành tạo dữ liệu đồ thị, chúng ta sẽ chuyển sang **Phase 3**:
1.  **Baseline Text/Image:** Train BERT/ResNet trên dữ liệu gốc.
2.  **Propagation Model:** Train GNN trên dữ liệu Cascade vừa tạo.

