# Legacy Baseline & Research Archive Document

> **Đóng băng hiện trạng trước khi chuẩn hóa Portfolio**  
> **Git Tag:** `legacy-before-portfolio-cleanup`  
> **Git Branch:** `portfolio-cleanup` (forked from `Dev` at commit `b8e6ed1`)  
> **Thời điểm đóng băng:** 2026-05-06 / 2026-11-11

---

## 1. Dữ liệu Lịch sử & Hiện trạng (Snapshot Statistics)

### 1.1. Dataset chính: `data/all_merged_binary.jsonl`
- **Tổng số bản ghi:** 9,584 records
- **Phân bố Label Binary:**
  - `REAL`: 7,576 records (79.05%)
  - `FAKE`: 2,008 records (20.95%)
- **Phân bố Nguồn dữ liệu (`source_dataset`):**
  - `Reddit`: 6,117 records (63.83%)
  - `Fakeddit`: 2,135 records (22.28%)
  - `voz`: 1,084 records (11.31%)
  - `news`: 248 records (2.59%)
- **Phân bố Phân đoạn (Split):**
  - `train`: 8,771 records (Reddit: 6,117, Fakeddit: 1,601, voz: 846, news: 207)
  - `val`: 406 records (Fakeddit: 254, voz: 134, news: 18)
  - `test`: 407 records (Fakeddit: 280, voz: 104, news: 23)
- **Graph Repository:** `data/processed_graphs_v3/` (9,584 graph files dạng PyTorch Geometric `.pt`, mỗi file chứa node features 1793 chiều và ma trận kề cascade `edge_index`).

### 1.2. Mã băm SHA-256 các file trọng yếu
| File Path | SHA-256 Hash | Ghi chú |
| :--- | :--- | :--- |
| `data/all_merged_binary.jsonl` | `ba95493b30e4149df5dd92731bc3c143e78b2e447c46cc7b8e568ae28399b350` | Dataset 9,584 mẫu nhị phân |
| `models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt` | `fdcff30346187233fcbf780fda6d199adc8d31a1efe2b703486849adf6e4ab83` | Checkpoint Multimodal GNN V3 (CrossAttention + GAT) |
| `models/checkpoints/rerun_full_canonical_seed42/best_multimodal_cascade_binary_cross_attention.pt` | `3e7e9ac0e2cfa528e3df57be6d6917c42a66e1cc2b7fc345860bd0bae7ee42e8` | Checkpoint Multimodal GNN V4 (Missing Image Emb) |

---

## 2. Kết quả Đánh giá Hiện tại (Baseline Evaluation)

Được kiểm tra độc lập trên tập test (`data/all_merged_binary.jsonl`, split `test`, 407 samples):

| Checkpoint / Cấu hình | Test Accuracy | Test Binary F1 | Validation Acc | Validation F1 | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `best_multimodal_cascade_binary_cross_attention.pt` | **0.8796 (87.96%)** | **0.7932** | 0.8966 (89.66%) | 0.8056 | **Canonical GNN V3 Baseline** |
| `rerun_full_canonical_seed42/...` | **0.8919 (89.19%)** | **0.8151** | 0.8990 (89.90%) | 0.8225 | Đã qua missing-image embedding |
| CLIP-only V2 Classifier | *0.9000 (90.00%)* | *0.8772* | 0.9040 | 0.8810 | Baseline tĩnh (không dùng Graph Cascade) |

---

## 3. Các Mô hình & Đường chạy Lịch sử (Legacy Models & Pipelines)

Trước đợt chuẩn hóa này, repository tồn tại nhiều nhánh nghiên cứu chạy song song:

1. **Phân loại 6 nhãn (6-class classification - Fakeddit gốc):**
   - Nhãn: `true`, `mostly_true`, `half_true`, `barely_true`, `false`, `pants_on_fire`.
   - Pipeline phân loại 6 class gặp tình trạng mất cân bằng dữ liệu nghiêm trọng, F1 thấp và không phản ánh đúng bài toán phát hiện tin giả thực tế trong sản phẩm.

2. **Nhánh CLIP-Only Champion (Fakeddit v2):**
   - Chỉ dùng text/image embedding từ CLIP ViT-B/32, phân loại bằng MLP tầng sâu và threshold tuning (0.36).
   - Đạt độ chính xác 90% trên một lát cắt dữ liệu tĩnh, nhưng hoàn toàn bỏ qua đồ thị lan truyền (Cascade Graph) và comment tree vốn là đóng góp học thuật cốt lõi của đề tài.

3. **Nhiều biến thể GNN phân mảnh:**
   - GraphSAGE, GCN, GAT thuần text (`src/training/train_gnn.py`).
   - Nhiều thư mục đồ thị: `data/04_graph/`, `data/processed_graphs/`, `data/processed_graphs_multimodal/`, `data/processed_graphs_v3/`.
   - Nhiều thư mục checkpoint: `models/checkpoints/auto_pipeline/`, `team_runs/`, `seed122_w24_exact/`, `rerun_full_seed...`.

4. **Vấn đề gán nhãn tự động dữ liệu Tiếng Việt (`src/data/import_vi_data.py`):**
   - File này trước đây gán mặc định toàn bộ bài từ diễn đàn Voz là `FAKE` và báo chí VnExpress là `REAL`. Đây là heuristic thiếu căn cứ ground-truth chặt chẽ, tạo ra domain leakage.

---

## 4. Lý do Chuyển đổi sang Canonical Pipeline Duy nhất

1. **Tính rõ ràng của sản phẩm (Clear Product Identity):**
   - Repository cần thể hiện một giải pháp nhất quán, có kiến trúc mạch lạc: **End-to-End Multimodal Cascade GNN Pipeline**.
2. **Loại bỏ phân mảnh:**
   - Khóa chặt duy nhất một format đồ thị (`processed_graphs_v3`), một bộ nhãn nhị phân chuẩn hóa (`REAL=0`, `FAKE=1`), và một checkpoint chính thức.
3. **Khả năng tái lập 100% (Reproducibility):**
   - Mọi kết quả công bố trên README và Demo phải được sinh ra từ file metrics có thể chạy lại từ code, không hardcode hay dùng heuristic biểu diễn.
4. **An toàn bảo mật:**
   - Dọn sạch các tệp cấu hình chứa key, token, service-account ra khỏi repo công khai và tài liệu hướng dẫn.

