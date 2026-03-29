# 🧪 Fake News Detection — Web Demo

Dashboard trực quan để chạy toàn bộ pipeline phát hiện tin giả: từ thu thập dữ liệu, xử lý, đến huấn luyện model.

---

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone repo

```bash
git clone https://github.com/<your-org>/Fake-New.git
cd Fake-New
```

### Bước 2: Tạo môi trường Conda

```bash
conda env create -f environment.yml
conda activate multimodal_gnn
```

Cài thêm Flask (chưa có trong `environment.yml`):

```bash
pip install flask
```

### Bước 3: Giải nén dữ liệu

Tải file dữ liệu từ Google Drive (link được gửi trong nhóm) và giải nén **vào thư mục gốc** của project:

```bash
# Giải nén file ZIP vào đúng thư mục project
# Ví dụ: unzip fakenews_data_minimal_20260309.zip -d .
# Trên Windows: Click phải → Extract Here
```

Sau khi giải nén, cấu trúc thư mục phải giống như sau:

```
Project/
├── data/
│   ├── reddit_enriched_data.jsonl        ← BẮT BUỘC
│   ├── cascade_visualization.html
│   ├── 01_raw/reddit/reddit_realtime_data.jsonl  ← BẮT BUỘC
│   ├── processed_graphs/                 ← Cần cho Training GNN
│   │   └── *.pt files
│   └── processed_graphs_multimodal/      ← Cần cho Training Multimodal
│       └── *.pt files
├── models/
│   └── checkpoints/                      ← Model đã train sẵn (tuỳ chọn)
│       └── *.pt files
├── demo-website/                         ← Web Dashboard
│   ├── app.py
│   ├── index.html
│   ├── script.js
│   ├── styles.css
│   └── run_history.py
└── src/                                  ← Source code pipeline
``` 

### Bước 4: Chạy Web Demo

```bash
cd demo-website
python app.py
```

Mở trình duyệt tại: **http://localhost:5000**

---

## 📋 Các chức năng trên Dashboard

| Mục | Chức năng | Mô tả |
|-----|-----------|-------|
| 🕷️ | Crawl Reddit | Thu thập bài viết từ Reddit |
| 💬 | Enrich Data | Làm giàu dữ liệu (comment trees) |
| 🔗 | Build Graphs | Tạo cascade graphs (PyG `.pt`) |
| 👁️ | Visualize | Tạo trang trực quan đồ thị |
| 🔄 | Reddit Pipeline | Xử lý 4 bước: Image→Text→Validate→LS |
| 📦 | Fakeddit Batch | Xử lý batch Fakeddit |
| 🏷️ | LS Convert | Chuyển Label Studio export → JSONL |
| 🤖 | Auto-Label | Tự động gán nhãn bằng AI |
| 🏋️ | Training | Train 5 loại model (Text/Image/Fusion/GNN/Multimodal) |
| 📊 | Results | Biểu đồ so sánh hiệu suất |
| 📜 | Run History | Lịch sử tất cả các lần chạy |

---

## 🔧 Cấu hình Python Path

File `app.py` sử dụng Python executable tại:
```
D:\anacoda3\envs\multimodal_gnn\python.exe
```

Nếu đường dẫn Conda của bạn khác, sửa dòng này trong `app.py`:
```python
PYTHON_EXE = sys.executable  # Tự động detect
```

---

## ❓ Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `ModuleNotFoundError: flask` | Chạy `pip install flask` |
| `UnicodeEncodeError` | Đã xử lý trong code, nếu vẫn lỗi chạy: `set PYTHONIOENCODING=utf-8` |
| Stats hiển thị 0 | Kiểm tra file `data/reddit_enriched_data.jsonl` có tồn tại không |
| Training lỗi "No graphs" | Cần giải nén `data/processed_graphs/` từ file ZIP |
| Port 5000 bị chiếm | Đổi port trong `app.py`: `app.run(port=5001)` |

---

## 👥 Cho thành viên muốn truy cập từ xa

Sửa dòng cuối trong `app.py` để mở trên mạng LAN:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

Sau đó thành viên cùng mạng WiFi truy cập bằng IP máy chủ: `http://<IP-máy-bạn>:5000`

---

## 📊 Kết nối với Binary Results Dashboard

Demo website hiện đã đọc trực tiếp các artifact trong `results/paper_figures/`. Để phần **Results** trên web hiển thị đúng Binary Accuracy / Binary F1 và các biểu đồ tăng giảm giữa các công nghệ xử lý, chạy thêm bước này sau khi có file kết quả experiment:

```bash
python src/experiments/visualize_results.py --results results/paper_experiments/<ten_file_ket_qua>.json
```

Khi đó web sẽ tự đọc:

- `results/paper_figures/binary_results_summary.md`
- `results/paper_figures/binary_percentage_comparison.png`
- `results/paper_figures/binary_accuracy_delta_vs_best_baseline.png`
- `results/paper_figures/binary_f1_delta_vs_best_baseline.png`
- `results/paper_figures/ablation_binary_delta_vs_full_model.png`
- `results/paper_figures/ablation_binary_f1_delta_vs_full_model.png`

Nếu chưa sinh các file này, dashboard vẫn mở bình thường nhưng phần kết quả sẽ không có summary/hình mới.

Ngoài ra, các nút pipeline chính trên web hiện đã được nối sang flow mới:

- `Reddit Pipeline` -> chuẩn bị batch Reddit cho Label Studio
- `LS Export -> JSONL` -> merge export và refresh file binary
- `Enrich Data` -> enrich comment tree và refresh `reddit_enriched_binary.jsonl`
- `Build Graphs` -> build graph text + multimodal
- `Training` -> train trên metadata binary hiện tại
