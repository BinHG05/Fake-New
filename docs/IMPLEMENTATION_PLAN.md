# 🚀 Phase 2 Implementation Plan: Cascade Data Collection & Graph Building

Tài liệu này tổng hợp **đánh giá code**, **các bước chạy**, và **kết quả kỳ vọng** cho toàn bộ Phase 2.

Ngày đánh giá: 2026-02-12

---

## 📊 Đánh giá tổng quan các file

### 1. `src/data/reddit_crawler.py` (305 dòng) — ✅ Đạt

| Tiêu chí | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| Crawl bài viết mới từ Reddit | ✅ | Lấy từ 5 subreddits: worldnews, news, politics, technology, conspiracy |
| Lấy comment tree (Cascade) | ✅ | `fetch_comments()` + `parse_comment_tree()` (DFS đệ quy) |
| Chuẩn hóa `user_id` | ✅ | Regex validate, fallback `unknown_user` |
| Chuẩn hóa `raw_text` | ✅ | Strip unicode, loại `[deleted]`, giới hạn 2000 ký tự |
| Phân loại `media_url` | ✅ | Chỉ giữ link media trực tiếp (jpg, png, mp4...) |
| Logging chuyên nghiệp | ✅ | RotatingFileHandler, tách `crawl.log` và `error.log` |
| Retry & Rate Limit | ✅ | `HTTPAdapter` + `Retry` strategy, `time.sleep` giữa các request |
| Chống trùng lặp | ✅ | `get_existing_ids()` kiểm tra trước khi lưu |
| Đường dẫn tuyệt đối | ✅ | Dùng `os.path.abspath(__file__)` |

**Nhận xét:** File này hoàn chỉnh, code production-ready. Không cần sửa gì thêm.

---

### 2. `src/data/crawler_enrich.py` (142 dòng) — ✅ Đạt

| Tiêu chí | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| Đọc file `labeled_master.jsonl` | ✅ | Từ `data/03_clean/Fakeddit/` |
| Gọi `fetch_comments()` lấy cascade | ✅ | Tái sử dụng logic từ `RedditCrawler` (DRY) |
| Gộp cascade vào dữ liệu gốc | ✅ | Thêm trường `cascade` và `metadata_enrich` |
| Lưu file kết quả (append mode) | ✅ | Lưu ngay sau mỗi item, an toàn khi bị crash |
| Resume (chạy tiếp) | ✅ | `get_existing_ids()` kiểm tra output, skip bài đã làm |
| Safe import | ✅ | `sys.path.append` + try/except fallback |
| Đường dẫn tuyệt đối | ✅ | Dùng `ROOT_DIR` tính từ vị trí file |
| Error handling | ✅ | try/except cho từng bài, không dừng cả chương trình |
| Rate Limit | ✅ | `time.sleep(2.0)` sau mỗi request |
| Encoding Windows | ✅ | `chcp 65001` fix font tiếng Việt |

**Nhận xét:** Đáp ứng đầy đủ yêu cầu "Cách B (Tách file)" trong Phase 2 Dev Guide. Code sạch, có resume, có error handling.

---

### 3. `build_final_graphs.py` (40 dòng) — ✅ Đạt

| Tiêu chí | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| Import `CascadeGraphBuilder` | ✅ | Từ `src.features.cascade_graph_builder` |
| Đọc file enriched data | ✅ | Từ `data/reddit_enriched_data.jsonl` |
| Gọi `process_dataset()` | ✅ | Biến text → embedding, comment → edge |
| Lưu file `.pt` | ✅ | Mỗi bài viết 1 file trong `data/processed_graphs/` |
| Resume | ✅ | Kiểm tra `os.path.exists(save_path)` trước khi lưu |

**Nhận xét:** Script ngắn gọn, đúng chức năng. Phụ thuộc vào `CascadeGraphBuilder` (đã kiểm tra, hoạt động đúng) và `TextEmbeddingExtractor` (XLM-RoBERTa, 768 chiều).

---

## 🔗 Sơ đồ luồng dữ liệu (Data Flow)

```
labeled_master.jsonl                  reddit_enriched_data.jsonl           processed_graphs/
(Dữ liệu đã gán nhãn)                (Dữ liệu + Comment Tree)            (Đồ thị .pt)
         │                                      │                               │
         ▼                                      ▼                               ▼
┌─────────────────────┐               ┌──────────────────────┐         ┌──────────────────┐
│ crawler_enrich.py   │ ───────────>  │ build_final_graphs.py│ ──────> │ post_id_1.pt     │
│                     │               │                      │         │ post_id_2.pt     │
│ Gọi RedditCrawler   │               │ CascadeGraphBuilder  │         │ ...              │
│ .fetch_comments()   │               │ + EmbeddingExtractor  │         │ post_id_N.pt     │
└─────────────────────┘               └──────────────────────┘         └──────────────────┘
     Bước 1                                  Bước 2                       Kết quả cuối
```

---

## 🛠️ Các bước chạy (Execution Steps)

### Bước 0: Kiểm tra điều kiện tiên quyết
```bash
# Kiểm tra file input có tồn tại không
dir data\03_clean\Fakeddit\labeled_master.jsonl

# Kiểm tra các thư viện cần thiết
pip install requests torch torch_geometric transformers networkx tqdm
```

**Kết quả kỳ vọng:** File `labeled_master.jsonl` tồn tại, các thư viện cài đặt thành công.

---

### Bước 1: Làm giàu dữ liệu (Data Enrichment)
```bash
python src/data/crawler_enrich.py
```

**Script thực hiện:**
1. Đọc `data/03_clean/Fakeddit/labeled_master.jsonl` (dữ liệu đã gán nhãn)
2. Với mỗi bài viết → Gọi Reddit API lấy toàn bộ comment tree
3. Gộp comment tree vào trường `cascade` của bài viết
4. Lưu kết quả vào `data/reddit_enriched_data.jsonl`

**Kết quả kỳ vọng:**
- File `data/reddit_enriched_data.jsonl` được tạo ra
- Mỗi dòng JSON chứa cấu trúc:
```json
{
  "id": "abc123",
  "raw_text": "Nội dung bài viết...",
  "label": "...",
  "cascade": [
    {"id": "cmt_1", "parent_id": "abc123", "user_id": "user_A", "timestamp": 1234567890, "text": "Comment 1", "level": 1},
    {"id": "cmt_2", "parent_id": "cmt_1", "user_id": "user_B", "timestamp": 1234567999, "text": "Reply to comment 1", "level": 2}
  ],
  "metadata_enrich": {
    "enriched_at": 1234568000,
    "comment_count_fetched": 2
  }
}
```

**Lưu ý quan trọng:**
- ⏱️ Tốn thời gian do Rate Limit (~2 giây/bài). 1000 bài ≈ 33 phút.
- 🔄 Có thể tắt/bật script thoải mái, nó sẽ tự chạy tiếp (Resume).
- ⚠️ Nếu gặp lỗi 429 (Too Many Requests), script tự xử lý, chờ rồi tiếp tục.

---

### Bước 2: Xây dựng đồ thị (Graph Building)
```bash
python build_final_graphs.py
```

**Script thực hiện:**
1. Đọc `data/reddit_enriched_data.jsonl`
2. Khởi tạo `CascadeGraphBuilder` (tự động load model XLM-RoBERTa)
3. Với mỗi bài viết:
   - Mã hóa nội dung bài viết + comment thành vector 768 chiều (`x`)
   - Xây dựng cấu trúc cạnh ai-trả-lời-ai (`edge_index`)
   - Đóng gói thành PyG `Data` object
4. Lưu mỗi bài thành file `.pt` trong `data/processed_graphs/`

**Kết quả kỳ vọng:**
- Thư mục `data/processed_graphs/` chứa các file `.pt`
- Mỗi file `.pt` chứa:

| Trường | Ý nghĩa | Shape |
| :--- | :--- | :--- |
| `x` | Node features (embedding vector) | `[N, 768]` (N = số node) |
| `edge_index` | Cạnh nối các node | `[2, E]` (E = số cạnh) |
| `post_id` | ID bài viết gốc | string |
| `num_nodes` | Số lượng node | int |

**Lưu ý quan trọng:**
- 📥 Lần đầu chạy sẽ tải model XLM-RoBERTa (~1GB). Cần mạng ổn định.
- 🖥️ Nếu có GPU (CUDA), script tự động sử dụng để tăng tốc.
- 🔄 Có resume: File `.pt` đã tồn tại sẽ bị bỏ qua.

---

### Bước 3: Kiểm tra kết quả (Verification)

```bash
# Kiểm tra file enriched data
dir data\reddit_enriched_data.jsonl

# Đếm số đồ thị đã tạo
dir data\processed_graphs\*.pt
```

```python
# Kiểm tra nội dung 1 file .pt
import torch
data = torch.load("data/processed_graphs/<post_id>.pt")
print(f"Số node: {data.num_nodes}")
print(f"Shape x: {data.x.shape}")        # Kỳ vọng: [N, 768]
print(f"Shape edge: {data.edge_index.shape}")  # Kỳ vọng: [2, E]
print(f"Post ID: {data.post_id}")
```

---

## ⚠️ Khắc phục sự cố thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
| :--- | :--- | :--- |
| `ModuleNotFoundError: reddit_crawler` | Chạy từ sai thư mục | Đảm bảo đang ở root project (`d:\NCKH_Project\Project`) |
| `FileNotFoundError: labeled_master.jsonl` | File input chưa có | Kiểm tra path `data/03_clean/Fakeddit/` |
| HTTP 429 (Rate Limit) | Reddit chặn do gọi API quá nhanh | Script tự xử lý, nếu vẫn lỗi thì đổi mạng hoặc chờ 10 phút |
| `CUDA out of memory` | GPU không đủ RAM | Thêm `--cpu` hoặc giảm batch size trong `embedding_extractor.py` |
| File `.pt` rỗng / thiếu | Bài viết không có comment | Bình thường, graph chỉ có 1 node (root) |

---

## ⏭️ Bước tiếp theo: Phase 3 (Baseline Training)

Sau khi hoàn thành tạo dữ liệu đồ thị, chuyển sang **Phase 3**:
1. **Baseline Text/Image:** Train BERT/ResNet trên dữ liệu gốc.
2. **Propagation Model:** Train GNN trên dữ liệu Cascade vừa tạo.
