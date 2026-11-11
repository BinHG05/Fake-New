# 🏆 KẾ HOẠCH CẢI TIẾN — ĐẠT GIẢI NGHIÊN CỨU KHOA HỌC

## CHIẾN LƯỢC TỔNG THỂ

Từ bộ Q&A 70 câu, tôi rút ra **12 điểm yếu có thể biến thành điểm mạnh** bằng code.
Mỗi mục dưới đây được đánh giá theo 3 tiêu chí:

- **Ảnh hưởng giải thưởng** (Impact): Giám khảo NCKH quan tâm nhất điều gì?
- **Độ khó triển khai** (Effort): Bao lâu để code xong?
- **Tính khả thi** (Feasible): Có chạy được trên máy hiện tại không?

---

## TIER 1 — LÀM NGAY (1-2 ngày, impact cao nhất)

### 1.1 ✏️ Thêm vùng "UNCERTAIN" cho output dự đoán
> **Vấn đề (Q&A 9.11):** Output hiện tại bắt buộc REAL/FAKE nhị phân.
> Giám khảo sẽ hỏi: "Bạn gắn nhãn FAKE mà không fact-check, liệu có gây hại?"

**Giải pháp:**
- Thêm zone `UNCERTAIN` khi `|P(REAL) - P(FAKE)| < threshold` (mặc định 0.15).
- Sửa `_predict_graph()` trong `predictor.py`:
  ```python
  margin = abs(real_prob - fake_prob)
  if margin < self.uncertain_threshold:
      prediction = "UNCERTAIN"
      confidence = max(real_prob, fake_prob)
  ```
- Sửa frontend hiển thị badge vàng "Cần kiểm chứng thêm" khi UNCERTAIN.

**File cần sửa:**
- `src/deployment/predictor.py` — thêm `uncertain_threshold` param + logic 3 nhãn.
- `demo-website/app.py` — frontend hiển thị UNCERTAIN.
- `demo-website/templates/` — CSS badge vàng.

**Ảnh hưởng:** ⭐⭐⭐⭐⭐ — Chứng minh nhận thức đạo đức AI (Responsible AI), rất ăn điểm NCKH.
**Effort:** ~2 giờ.

---

### 1.2 📊 Thống kê Topology Graph: REAL vs FAKE
> **Vấn đề (Q&A 5.3):** Claim "tin giả có nhánh cãi vã sâu" nhưng chưa có bằng chứng thống kê.

**Giải pháp:** Viết script phân tích trên toàn bộ 2384 graphs:
- Tính: `depth`, `breadth`, `branching_factor`, `avg_degree` cho REAL vs FAKE.
- Vẽ biểu đồ boxplot so sánh.
- Chạy Mann-Whitney U test kiểm định thống kê (p-value).
- Lưu vào `results/paper_figures/graph_topology_comparison.png`.

**File cần tạo:**
- `scripts/analyze_graph_topology.py`
- Output: `results/paper_figures/topology_real_vs_fake.png` + `topology_stats.json`

**Ảnh hưởng:** ⭐⭐⭐⭐⭐ — Biến nhận định chủ quan thành **bằng chứng khoa học có p-value**.
**Effort:** ~3 giờ.

---

### 1.3 🧪 Kiểm tra Keyword Leakage
> **Vấn đề (Q&A 5.7):** Comment chứa "fake", "hoax" → model có thể học shortcut.

**Giải pháp:** Tạo script kiểm chứng:
1. Load test set graphs.
2. Với mỗi graph, mask tất cả node features có chứa từ khóa bóc phốt ("fake", "hoax", "source", "debunked") bằng cách thay embedding = mean embedding.
3. Chạy inference lại trên masked test set.
4. So sánh accuracy/F1 trước và sau mask.
5. Nếu delta < 5 pp → Kết luận: "Model không phụ thuộc shortcut keyword."

**File cần tạo:**
- `scripts/test_keyword_leakage.py`
- Output: `results/keyword_leakage_report.json`

**Ảnh hưởng:** ⭐⭐⭐⭐⭐ — Trực tiếp loại bỏ nghi ngờ lớn nhất về tính valid của mô hình.
**Effort:** ~3 giờ.

---

### 1.4 🔥 Temperature Scaling (Probability Calibration)
> **Vấn đề (Q&A 6.8):** Output softmax chưa calibrate — 63% không thực sự = "63% chắc chắn".

**Giải pháp:**
1. Sau khi train xong best model, chạy Temperature Scaling trên validation set.
2. Tìm optimal temperature `T` bằng NLL minimization.
3. Lưu `T` vào checkpoint.
4. Inference: `probs = softmax(logits / T)` thay vì `softmax(logits)`.
5. Vẽ **Reliability Diagram** (calibration curve) trước/sau.

**File cần tạo/sửa:**
- `scripts/calibrate_temperature.py`
- `src/deployment/predictor.py` — thêm `self.temperature` param.
- Output: `results/paper_figures/calibration_curve.png`

**Ảnh hưởng:** ⭐⭐⭐⭐ — Chứng minh hiểu sâu về probabilistic inference, rất ấn tượng.
**Effort:** ~2 giờ.

---

## TIER 2 — LÀM TRONG TUẦN (2-3 ngày, impact trung-cao)

### 2.1 🧠 Learned [MISSING] Embedding thay thế Zero-Padding
> **Vấn đề (Q&A 4.3, 9.3):** Zero-padding image khi thiếu ảnh → model có thể học bias.

**Giải pháp:**
- Thêm `self.missing_image_emb = nn.Parameter(torch.randn(image_dim))` vào `MultimodalCascadeGNN`.
- Khi image feature = 0 vector → thay bằng `missing_image_emb`.
- Train lại model (50 epochs, ~1-2 giờ GPU).
- So sánh kết quả với zero-padding → Ghi vào ablation study.

**File cần sửa:**
- `src/training/train_multimodal_gnn.py` — thêm `missing_image_emb` vào model.
- `src/deployment/predictor.py` — inference dùng missing embedding khi load model mới.

**Ảnh hưởng:** ⭐⭐⭐⭐ — Cải thiện robustness, có thể tăng accuracy.
**Effort:** ~4 giờ (bao gồm retrain).

---

### 2.2 🌍 Domain Shift Evaluation (Out-of-Distribution Test)
> **Vấn đề (Q&A 2.5):** Chưa có đánh giá formal khi test ngoài Fakeddit.

**Giải pháp:**
1. Crawl 50 bài từ các subreddit mới KHÔNG có trong tập train:
   - 25 bài "tin thật": r/science, r/worldnews, r/neutralnews
   - 25 bài "tin giả": r/TheOnion, r/AteTheOnion, r/satire
2. Gán nhãn thủ công (ground truth).
3. Chạy predict_url() trên 50 bài → Tính accuracy, F1.
4. So sánh với in-distribution test set → Đo mức domain shift.

**File cần tạo:**
- `scripts/eval_domain_shift.py`
- `data/ood_test_set.jsonl` — 50 bài đã gán nhãn.
- Output: `results/domain_shift_evaluation.json`

**Ảnh hưởng:** ⭐⭐⭐⭐⭐ — Trực tiếp trả lời câu hỏi "model có generalize được không?"
**Effort:** ~5 giờ (crawl mất thời gian vì rate limit).

---

### 2.3 📈 Explainability — Attention Visualization
> **Vấn đề (Q&A 9.6):** Không giải thích được vì sao dự đoán FAKE.

**Giải pháp:**
1. Thêm option `return_attention=True` khi forward pass GAT.
2. Thu thập attention weights từ `GATConv`.
3. Vẽ heatmap: Node nào được "chú ý" nhiều nhất → Bình luận nào quyết định kết quả.
4. Hiển thị trên frontend: Highlight top-3 comment ảnh hưởng lớn nhất.

**File cần sửa:**
- `src/training/train_multimodal_gnn.py` — GATConv return attention.
- `src/deployment/predictor.py` — thu thập attention weights.
- `demo-website/app.py` — API trả thêm `top_influential_comments`.
- `demo-website/templates/` — UI hiển thị.

**Ảnh hưởng:** ⭐⭐⭐⭐⭐ — XAI (Explainable AI) là xu hướng nóng nhất hiện nay. Demo có cái này = ăn điểm nặng.
**Effort:** ~6 giờ.

---

### 2.4 ⏱️ Temporal Positional Encoding cho Comment
> **Vấn đề (Q&A 5.2):** Timestamp lưu nhưng không dùng.

**Giải pháp:**
- Tính `delta_t = comment_time - post_time` cho mỗi node.
- Encode bằng sinusoidal positional encoding (giống Transformer):
  ```python
  pe = sin(delta_t / 10000^(2i/d_model))
  ```
- Concat `pe` (16 chiều) vào node feature trước fusion.
- Train lại + so sánh ablation with/without temporal.

**File cần sửa:**
- `src/features/cascade_graph_builder.py` — thêm temporal encoding.
- `src/training/train_multimodal_gnn.py` — adjust `input_dim`.

**Ảnh hưởng:** ⭐⭐⭐ — Novelty khoa học, nhưng gain thực tế có thể nhỏ trên dataset này.
**Effort:** ~5 giờ.

---

## TIER 3 — LÀM NẾU CÒN THỜI GIAN (Nâng tầm paper)

### 3.1 🛡️ Adversarial Robustness Test
> **Vấn đề (Q&A 5.8):** Spam "fake" vào bài thật → model bị thao túng?

**Giải pháp:**
1. Lấy 50 bài REAL từ test set.
2. Inject 5-10 comment giả chứa "fake", "hoax", "bullshit" vào cascade.
3. Chạy lại inference → Đo tỷ lệ bị flip sang FAKE.
4. Nếu flip rate < 20% → "Model có mức resilience chấp nhận được."

**File cần tạo:** `scripts/test_adversarial_robustness.py`
**Ảnh hưởng:** ⭐⭐⭐⭐ — Chứng minh AI safety awareness.
**Effort:** ~4 giờ.

---

### 3.2 📊 Cross-Domain Comparison Table
> **Vấn đề:** Chưa so sánh với state-of-the-art papers khác.

**Giải pháp:** Tạo bảng so sánh với các phương pháp trong literature:
| Method | Dataset | Binary Acc | F1 |
|--------|---------|-----------|-----|
| dEFEND (Shu et al., 2019) | FakeNewsNet | 92.0% | — |
| BiGCN (Bian et al., 2020) | Twitter15/16 | 88.3% | 86.7% |
| UPFD (Dou et al., 2021) | FakeNewsNet | 87.1% | — |
| **Ours (Multimodal GNN)** | **Fakeddit** | **66.4%** | **56.9%** |

- Ghi rõ: "So sánh tham khảo, các phương pháp trên dùng dataset/metric khác nhau."
- Mục đích: Positioning đề tài trong bối cảnh nghiên cứu rộng hơn.

**File cần tạo:** `results/cross_domain_comparison.md`
**Ảnh hưởng:** ⭐⭐⭐⭐ — Chứng minh survey literature có chiều sâu.
**Effort:** ~2 giờ (chỉ cần đọc paper và ghi số).

---

### 3.3 🔄 Retrain với Data Augmentation
> **Vấn đề:** Dataset chỉ 2384 mẫu, khá nhỏ.

**Giải pháp:**
- **Graph augmentation:** Random drop nodes (10-20%), random drop edges (10-20%), feature masking.
- Thêm vào DataLoader transform.
- Retrain → kỳ vọng tăng 2-5 pp accuracy + giảm overfitting.

**Ảnh hưởng:** ⭐⭐⭐ — Tăng hiệu năng thực tế.
**Effort:** ~4 giờ.

---

### 3.4 📄 Missing Modality Inference Test
> **Vấn đề (Q&A 9.5):** Fallback chưa được test chính thức.

**Giải pháp:**
1. Load test set, chia 3 kịch bản:
   - (a) Full features (bình thường).
   - (b) Zero-out image features (giả lập mất ảnh).
   - (c) Chỉ giữ root node, xóa comments (giả lập mất comment).
2. Chạy inference 3 kịch bản → So sánh accuracy.
3. Vẽ bar chart comparison.

**File cần tạo:** `scripts/test_missing_modality.py`
**Ảnh hưởng:** ⭐⭐⭐⭐ — Biến "thiết kế" thành "bằng chứng thực nghiệm".
**Effort:** ~3 giờ.

---

## 📋 CHECKLIST ƯU TIÊN — THỨ TỰ TRIỂN KHAI

Sắp xếp theo công thức: **Impact × Feasibility / Effort**

| # | Task | Impact | Effort | Ưu tiên |
|---|------|--------|--------|---------|
| 1 | UNCERTAIN zone (1.1) | ⭐⭐⭐⭐⭐ | 2h | 🔴 Làm đầu tiên |
| 2 | Keyword Leakage Test (1.3) | ⭐⭐⭐⭐⭐ | 3h | 🔴 Làm đầu tiên |
| 3 | Graph Topology Stats (1.2) | ⭐⭐⭐⭐⭐ | 3h | 🔴 Làm đầu tiên |
| 4 | Temperature Scaling (1.4) | ⭐⭐⭐⭐ | 2h | 🟡 Làm ngay sau Tier 1 |
| 5 | Domain Shift Eval (2.2) | ⭐⭐⭐⭐⭐ | 5h | 🟡 Quan trọng nhất Tier 2 |
| 6 | Explainability/XAI (2.3) | ⭐⭐⭐⭐⭐ | 6h | 🟡 WOW factor cao nhất |
| 7 | Missing Modality Test (3.4) | ⭐⭐⭐⭐ | 3h | 🟡 Chuyển thiết kế → bằng chứng |
| 8 | Learned [MISSING] Emb (2.1) | ⭐⭐⭐⭐ | 4h | 🟢 Nếu còn thời gian |
| 9 | Adversarial Test (3.1) | ⭐⭐⭐⭐ | 4h | 🟢 Nếu còn thời gian |
| 10 | Cross-Domain Table (3.2) | ⭐⭐⭐⭐ | 2h | 🟢 Dễ, làm nhanh |
| 11 | Temporal Encoding (2.4) | ⭐⭐⭐ | 5h | 🔵 Optional |
| 12 | Data Augmentation (3.3) | ⭐⭐⭐ | 4h | 🔵 Optional |

**Tổng effort ước tính:**
- 🔴 Tier 1 (bắt buộc): ~10 giờ
- 🟡 Tier 2 (nên làm): ~16 giờ
- 🟢🔵 Tier 3 (bonus): ~15 giờ

---

## 🎯 SAU KHI HOÀN THÀNH — BẠN SẼ CÓ GÌ?

1. **Responsible AI:** Vùng UNCERTAIN + ngưỡng calibrate → "Hệ thống biết khi nào nó không chắc chắn."
2. **Khoa học chặt chẽ:** Topology stats có p-value + Keyword leakage test + Domain shift eval.
3. **Explainability:** Attention heatmap trên graph → "Bình luận số 3 và 7 là lý do model phán FAKE."
4. **Robustness đã kiểm chứng:** Missing modality test + Adversarial test → Bằng chứng thực nghiệm.
5. **So sánh quốc tế:** Cross-domain table → Định vị trong literature.

→ Một project có ĐẦY ĐỦ các yếu tố trên sẽ vượt mặt bằng chung NCKH sinh viên rất xa.
