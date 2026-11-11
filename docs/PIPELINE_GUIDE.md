# 🚀 HƯỚNG DẪN CHẠY PIPELINE DỮ LIỆU

Tài liệu này hướng dẫn **toàn bộ quy trình** từ dữ liệu thô đến đồ thị sẵn sàng train GNN.
Bất kỳ ai trong nhóm cũng có thể đọc và tự chạy được.

---

## 📋 Yêu cầu trước khi chạy

### Cài đặt thư viện
```bash
pip install torch torch_geometric transformers networkx tqdm requests
```

### Cấu trúc folder cần có
```
Project/
├── data/
│   ├── 03_clean/Fakeddit/labeled_master.jsonl   ← Dữ liệu đầu vào (đã label)
│   ├── reddit_enriched_data.jsonl               ← Sẽ được tạo ở Bước 1
│   ├── processed_graphs/                        ← Sẽ được tạo ở Bước 2
│   └── cascade_visualization.html               ← Sẽ được tạo ở Bước 3
├── src/data/crawler_enrich.py
├── build_final_graphs.py
└── visualize_graphs.py
```

---

## 🔄 Quy trình 3 bước

### Bước 1️⃣ — Làm giàu dữ liệu (Lấy comment cascade từ Reddit)

```bash
python src/data/crawler_enrich.py
```

| Mục | Chi tiết |
|-----|----------|
| **Đầu vào** | `data/03_clean/Fakeddit/labeled_master.jsonl` |
| **Đầu ra** | `data/reddit_enriched_data.jsonl` |
| **Làm gì?** | Đọc từng bài viết → Gọi API Reddit → Lấy cây bình luận (cascade) → Gộp vào dữ liệu |
| **Thời gian** | ~2-5 giây/bài (do rate limit API Reddit) |
| **Resume?** | ✅ Có. Bài nào đã xử lý sẽ tự động bỏ qua |

---

### Bước 2️⃣ — Xây dựng đồ thị (.pt files)

```bash
python build_final_graphs.py
```

| Mục | Chi tiết |
|-----|----------|
| **Đầu vào** | `data/reddit_enriched_data.jsonl` |
| **Đầu ra** | Folder `data/processed_graphs/` chứa các file `.pt` |
| **Làm gì?** | Đọc dữ liệu đã làm giàu → Mã hóa text thành vector bằng XLM-RoBERTa → Xây cây đồ thị → Lưu thành `.pt` |
| **Thời gian** | Lần đầu chạy sẽ tải model (~500MB). Sau đó ~1-3 giây/bài |
| **Resume?** | ✅ Có. File `.pt` nào đã tồn tại sẽ tự động bỏ qua |

**Mỗi file `.pt` chứa:**
- `x` — Ma trận đặc trưng `[N, 768]` (N = số node, 768 = chiều embedding)
- `edge_index` — Ma trận cạnh `[2, E]` (E = số cạnh, mỗi cột là 1 cặp `[nguồn, đích]`)
- `post_id` — ID bài viết gốc

---

### Bước 3️⃣ — Trực quan hóa (Tùy chọn)

```bash
python visualize_graphs.py
```

| Mục | Chi tiết |
|-----|----------|
| **Đầu vào** | Folder `data/processed_graphs/` |
| **Đầu ra** | `data/cascade_visualization.html` |
| **Làm gì?** | Đọc tất cả file `.pt` → Tạo dashboard HTML tương tác để xem cấu trúc đồ thị |

**Cách dùng:** Mở file `cascade_visualization.html` trong trình duyệt → Bấm "👁️ Xem" để xem node nào nối với node nào.

---

## 🔁 Khi có dữ liệu mới

Mỗi khi `labeled_master.jsonl` được cập nhật thêm bài mới, chỉ cần **chạy lại đúng 3 lệnh** trên theo thứ tự:

```bash
python src/data/crawler_enrich.py      # Bước 1: Làm giàu
python build_final_graphs.py           # Bước 2: Xây đồ thị
python visualize_graphs.py             # Bước 3: Cập nhật visualization (tùy chọn)
```

> **💡 Lưu ý:** Cả 3 script đều có cơ chế **Resume** — chỉ xử lý bài mới, bỏ qua bài đã làm. Nên chạy lại rất nhanh và an toàn.

---

## ⚠️ Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Chưa cài PyTorch | `pip install torch` |
| `ConnectionError` ở Bước 1 | Mất mạng / Reddit chặn | Đợi vài phút rồi chạy lại (có Resume) |
| `CUDA out of memory` ở Bước 2 | GPU hết RAM | Script sẽ tự dùng CPU, hoặc set `CUDA_VISIBLE_DEVICES=""` |
| File `.pt` bị lỗi | Quá trình lưu bị gián đoạn | Xóa file `.pt` lỗi đó rồi chạy lại Bước 2 |

---

## 📊 Kiểm tra kết quả

Sau khi chạy xong, kiểm tra bằng Python:

```python
import torch
data = torch.load("data/processed_graphs/[post_id].pt", map_location='cpu', weights_only=False)
print(f"Số node: {data.num_nodes}")
print(f"Kích thước features: {data.x.shape}")      # [N, 768]
print(f"Kích thước edge_index: {data.edge_index.shape}")  # [2, E]
print(f"Post ID: {data.post_id}")
```

---

## 🌐 Chạy Demo Website

Sau khi đã có dữ liệu, graph và kết quả train/experiment, cả nhóm có thể mở web demo để theo dõi pipeline và xem kết quả trực quan.

### Bước 4 - Sinh hình kết quả cho demo

```bash
python src/experiments/visualize_results.py --results results/paper_experiments/<ten_file_ket_qua>.json
```

Lệnh này sẽ tạo các file trong `results/paper_figures/`, gồm:

- `binary_results_summary.md`
- `binary_percentage_comparison.png`
- `binary_accuracy_delta_vs_best_baseline.png`
- `binary_f1_delta_vs_best_baseline.png`
- `ablation_binary_delta_vs_full_model.png`
- `ablation_binary_f1_delta_vs_full_model.png`

### Bước 5 - Chạy demo website

```bash
cd demo-website
python app.py
```

Mở trình duyệt tại:

```text
http://localhost:5000
```

### Demo website sẽ hiển thị gì?

- trạng thái dữ liệu hiện tại
- các bước pipeline và log khi chạy
- lịch sử các lần train/crawl/enrich/build graph
- biểu đồ `Binary Accuracy` và `Binary F1`
- summary markdown và các hình kết quả trong `results/paper_figures/`

### Các nút chính trên demo hiện đang gọi flow nào?

- `Reddit Pipeline` -> `research_pipeline.py prepare-label`
- `LS Export -> JSONL` -> `research_pipeline.py merge-labels`
- `Enrich Data` -> `research_pipeline.py enrich`
- `Build Graphs` -> `research_pipeline.py build-graphs --multimodal`
- `Training` -> train trên metadata binary và graph binary hiện tại

### Checklist trước khi mở demo

- đã có `data/reddit_enriched_binary.jsonl` hoặc dữ liệu enrich tương ứng
- đã có graph trong `data/processed_graphs/` hoặc `data/processed_graphs_multimodal/`
- đã chạy `visualize_results.py` để sinh file trong `results/paper_figures/`
- đang dùng đúng môi trường Python có Flask và các thư viện của project

*Cập nhật lần cuối: 29/03/2026*
