# 📋 PHÂN CÔNG NHIỆM VỤ NHÓM NCKH
## Kế hoạch làm việc cho các thành viên

> **Cập nhật:** 25/04/2026
> **Trưởng nhóm (Leader):** [Tên bạn]
> **Mục tiêu:** Hoàn thành các thí nghiệm khoa học + cải tiến pipeline để bảo vệ giải NCKH

---

## 📊 TỔNG QUAN PHÂN CÔNG

| Vai trò | Người thực hiện | Nhóm nhiệm vụ chính |
|---------|----------------|---------------------|
| **Leader** | Bạn | Cải tiến model, chạy thí nghiệm, viết script phân tích |
| **Thành viên A** | _________ | Cào data Reddit mới + Gán nhãn |
| **Thành viên B** | _________ | Gán nhãn + Làm báo cáo / Slide |
| **Thành viên C** | _________ | Clean pipeline + Cải tiến hệ thống (IMPROVEMENT_PLAN) |

> [!NOTE]
> Các nhiệm vụ được sắp xếp theo **độ ưu tiên giảm dần**. Làm hết phần 🔴 trước khi chuyển sang 🟡.

---

## 🔴 GIAI ĐOẠN 1 — SONG SONG (Tuần 1)

### 👤 LEADER — Chạy 5 Thí Nghiệm Quyết Định

> ⚠️ **Phần này Leader tự thực hiện, KHÔNG phân công.**

| # | Thí nghiệm | Thời gian | Output |
|---|------------|-----------|--------|
| 1 | TN5: Cascade Contribution (Before/After) | 5h | `cascade_vs_no_cascade.png`, `accuracy_vs_cascade_size.png` |
| 2 | TN1: Ablation Table | 3h | `ablation_definitive.png`, `ablation_definitive.tex` |
| 3 | TN2: Error Analysis | 5h | `confusion_matrix.png`, `accuracy_by_graph_size.png` |
| 4 | TN3: Robustness Test | 6h | `robustness_*.png` |
| 5 | TN4: Domain Shift (phần script + inference) | 3h | `domain_shift_comparison.png` |

**Tổng: ~22h**

---

### 👤 THÀNH VIÊN A — Cào Data & Gán Nhãn

#### Nhiệm vụ A1: Cào data Reddit mới cho Domain Shift Test 🔴
**Deadline: Ngày 2 của Tuần 1**

**Mục tiêu:** Thu thập 50 bài Reddit MỚI hoàn toàn ngoài tập Fakeddit để test khả năng generalize của model.

**Hướng dẫn chi tiết:**

1. **Cào 25 bài TIN THẬT** từ các subreddit uy tín:
   - `r/science` — bài có nguồn dẫn, đã peer-review
   - `r/worldnews` — tin tức quốc tế từ nguồn chính thống
   - `r/neutralnews` — tin đã được verify

2. **Cào 25 bài TIN GIẢ** từ các subreddit satire/debunked:
   - `r/TheOnion` — satire rõ ràng
   - `r/nottheonion` — tin tức kỳ lạ nhưng thật (lưu ý gán nhãn đúng)
   - `r/AteTheOnion` — người tin vào tin giả
   - Hoặc bài đã bị fact-check debunk trên `r/quityourbullshit`

3. **Lệnh cào data:**
   ```powershell
   # Cào từ subreddit cụ thể
   python src/utils/research_pipeline.py crawl --limit 25 --images-only
   ```

4. **Tiêu chí chọn bài:**
   - ✅ Bài phải có ít nhất 1 ảnh đính kèm
   - ✅ Bài phải có ít nhất 3 comments
   - ✅ Bài không nằm trong tập Fakeddit gốc
   - ✅ Bài bằng tiếng Anh
   - ❌ KHÔNG lấy bài đã bị xóa hoặc [removed]
   - ❌ KHÔNG lấy bài chỉ có link, không có text

5. **Format output:**
   Mỗi bài cần ghi lại vào file `data/ood_test_set_raw.jsonl`:
   ```json
   {
     "id": "reddit_post_id",
     "subreddit": "r/science",
     "title": "...",
     "text": "...",
     "url": "https://reddit.com/r/...",
     "image_url": "...",
     "num_comments": 15,
     "source_type": "credible_news",
     "crawl_date": "2026-04-XX"
   }
   ```

**Deliverable:**
- [ ] File `data/ood_test_set_raw.jsonl` chứa 50 bài
- [ ] Spreadsheet tracking: subreddit, post_id, title, loại tin (thật/giả)

---

#### Nhiệm vụ A2: Gán nhãn cho data Domain Shift 🔴
**Deadline: Ngày 3 của Tuần 1**

**Mục tiêu:** Gán nhãn ground truth cho 50 bài đã cào.

**Tiêu chí gán nhãn (QUAN TRỌNG — phải tuân thủ nghiêm ngặt):**

| Nhãn | Tiêu chí |
|------|---------|
| **REAL** | Tin tức từ nguồn uy tín, có thể verify bằng ≥2 nguồn tin độc lập |
| **FAKE** | Tin satire, tin đã bị debunk, hoặc thông tin sai lệch có bằng chứng |

**Quy trình:**
1. Đọc title + text đầy đủ của bài
2. Kiểm tra nguồn gốc (trang web, tác giả)
3. Cross-check với ít nhất 1 nguồn khác (Google, fact-check sites)
4. Ghi rõ lý do gán nhãn vào cột `label_reason`

**Output:** File `data/ood_test_set.jsonl` — 50 bài đã gán nhãn REAL/FAKE:
```json
{
  "id": "reddit_post_id",
  "url": "...",
  "label": "REAL",
  "label_reason": "Tin từ Nature journal, đã verify trên Reuters",
  "labeled_by": "ThanhVienA",
  "label_date": "2026-04-XX"
}
```

**Deliverable:**
- [ ] File `data/ood_test_set.jsonl` — 50 bài đã gán nhãn
- [ ] Mỗi bài đều có `label_reason`
- [ ] Cross-check: ít nhất 10 bài được thành viên B kiểm tra lại

---

#### Nhiệm vụ A3: Cào thêm data bổ sung tập train 🟡
**Deadline: Cuối Tuần 2**

**Mục tiêu:** Bổ sung thêm 100-200 bài Reddit mới vào tập train để tăng size dataset.

**Lệnh:**
```powershell
# Cào bài có ảnh
python src/utils/research_pipeline.py crawl --limit 100 --images-only

# Tạo batch cho Label Studio
python src/utils/research_pipeline.py prepare-label
```

**Deliverable:**
- [ ] Cào được ≥100 bài mới
- [ ] Đã chạy `prepare-label` tạo batch cho Label Studio
- [ ] Đã import vào Label Studio để gán nhãn

---

### 👤 THÀNH VIÊN B — Gán Nhãn & Làm Báo Cáo

#### Nhiệm vụ B1: Hỗ trợ gán nhãn data Domain Shift 🔴
**Deadline: Ngày 3 của Tuần 1**

**Mục tiêu:** Cross-check nhãn của Thành viên A + gán nhãn data bổ sung.

**Công việc:**
1. Kiểm tra lại ≥10 bài mà Thành viên A đã gán nhãn → Ghi nhận agreement/disagreement
2. Với các bài disagreement → Thảo luận với A để thống nhất nhãn cuối cùng
3. Tính **Inter-annotator Agreement** (Cohen's Kappa) — báo lại cho Leader

**Deliverable:**
- [ ] Báo cáo cross-check: tỷ lệ đồng thuận (%)
- [ ] Cohen's Kappa score
- [ ] Danh sách bài bất đồng + nhãn cuối cùng sau thảo luận

---

#### Nhiệm vụ B2: Gán nhãn data bổ sung trên Label Studio 🟡
**Deadline: Tuần 2**

**Mục tiêu:** Gán nhãn cho batch data mới mà Thành viên A đã cào.

**Quy trình:**
1. Nhận file `reddit_for_ls.json` từ batch mới
2. Import vào Label Studio
3. Gán nhãn theo tiêu chí chuẩn (xem `docs/LABELING_GUIDE.md`)
4. Export JSON khi hoàn thành

**Lệnh merge sau khi export:**
```powershell
python src/utils/research_pipeline.py merge-labels --input "data/03_clean/Reddit/<batch_name>/export_<batch_name>.json"
```

**Deliverable:**
- [ ] Export JSON từ Label Studio
- [ ] Đã merge vào `labeled_master.jsonl`
- [ ] Báo cáo: số bài đã gán nhãn, phân bố REAL/FAKE

---

#### Nhiệm vụ B3: Cross-Domain Comparison Table 🟡
**Deadline: Tuần 2**

**Mục tiêu:** Tạo bảng so sánh kết quả của nhóm với các phương pháp state-of-the-art trong literature.

**Công việc:**
1. Đọc các paper sau và ghi lại kết quả:
   - dEFEND (Shu et al., 2019) — FakeNewsNet
   - BiGCN (Bian et al., 2020) — Twitter15/16
   - UPFD (Dou et al., 2021) — FakeNewsNet
   - SAFE (Zhou et al., 2020) — GossipCop
   - MCAN (Wu et al., 2021) — Twitter + Weibo
2. Tạo bảng markdown theo format:

| Method | Year | Dataset | Binary Acc | F1 | Graph? | Multimodal? |
|--------|------|---------|-----------|-----|--------|-------------|
| dEFEND | 2019 | FakeNewsNet | 92.0% | — | ✅ | ❌ |
| BiGCN | 2020 | Twitter15/16 | 88.3% | 86.7% | ✅ | ❌ |
| UPFD | 2021 | FakeNewsNet | 87.1% | — | ✅ | ❌ |
| **Ours** | **2026** | **Fakeddit** | **66.4%** | **56.9%** | **✅** | **✅** |

3. Ghi chú quan trọng: "So sánh mang tính tham khảo — các phương pháp sử dụng dataset và metric khác nhau."

**Deliverable:**
- [ ] File `results/cross_domain_comparison.md` — bảng so sánh hoàn chỉnh
- [ ] Ghi rõ citation cho mỗi paper (authors, year, venue)
- [ ] Phần nhận xét: điểm mạnh/yếu của phương pháp nhóm so với SOTA

---

#### Nhiệm vụ B4: Làm báo cáo / Slide bảo vệ 🟡
**Deadline: Tuần 3**

**Mục tiêu:** Soạn slide trình bày + báo cáo paper cho buổi bảo vệ.

**Cấu trúc slide đề xuất (15-20 slides):**

| Slide | Nội dung |
|-------|---------|
| 1 | Trang bìa: Tên đề tài, nhóm, trường |
| 2-3 | Bối cảnh & Vấn đề nghiên cứu |
| 4 | Mục tiêu & Câu hỏi nghiên cứu |
| 5-6 | Tổng quan phương pháp liên quan (Related Work) |
| 7-8 | Kiến trúc hệ thống (Pipeline diagram) |
| 9 | Dataset & Preprocessing |
| 10-11 | Kết quả Ablation Study (TN1) — dùng `ablation_definitive.png` |
| 12 | Cascade Contribution (TN5) — dùng `cascade_vs_no_cascade.png` |
| 13 | Error Analysis (TN2) — dùng `confusion_matrix.png` |
| 14 | Robustness Test (TN3) — dùng `robustness_*.png` |
| 15 | Domain Shift (TN4) — dùng `domain_shift_comparison.png` |
| 16 | Demo hệ thống (screenshots) |
| 17 | Kết luận & Đóng góp chính |
| 18 | Hạn chế & Hướng phát triển |
| 19 | Q&A |

**3 câu kết luận PHẢI có trong slide:**
1. "Cascade comment graph enrichment cải thiện performance +XX pp so với content-only."
2. "Cross-Attention fusion vượt trội Gated fusion +20 pp."
3. "Hệ thống duy trì XX% accuracy khi thiếu image modality."

**Deliverable:**
- [ ] File slide (.pptx hoặc Google Slides)
- [ ] Báo cáo paper (word/LaTeX) — draft đầu tiên
- [ ] Các figure từ `results/paper_figures/` đã được nhúng vào slide

---

### 👤 THÀNH VIÊN C — Clean Pipeline & Cải Tiến Hệ Thống

> Phần này tham chiếu từ `IMPROVEMENT_PLAN.md`

#### Nhiệm vụ C1: Thêm vùng UNCERTAIN cho output 🔴
**Deadline: Ngày 2 của Tuần 1**
**Effort: ~2 giờ**

**Mục tiêu:** Thêm nhãn thứ 3 `UNCERTAIN` khi model không chắc chắn, thay vì bắt buộc REAL/FAKE.

**File cần sửa:**

1. **`src/deployment/predictor.py`**:
   - Thêm parameter `uncertain_threshold = 0.15`
   - Sửa hàm `_predict_graph()`:
   ```python
   margin = abs(real_prob - fake_prob)
   if margin < self.uncertain_threshold:
       prediction = "UNCERTAIN"
       confidence = max(real_prob, fake_prob)
   ```

2. **`demo-website/app.py`**:
   - Sửa response để trả về nhãn UNCERTAIN
   - Hiển thị badge vàng "Cần kiểm chứng thêm" trên frontend

3. **`demo-website/templates/`**:
   - Thêm CSS cho badge vàng UNCERTAIN
   - Icon ⚠️ + text "Cần kiểm chứng thêm"

**Tiêu chí hoàn thành:**
- [ ] Khi `|P(REAL) - P(FAKE)| < 0.15` → hiển thị UNCERTAIN
- [ ] Frontend hiển thị đúng badge vàng
- [ ] Test: chạy 5 URL khác nhau, screenshot kết quả

---

#### Nhiệm vụ C2: Temperature Scaling (Calibration) 🔴
**Deadline: Ngày 3 của Tuần 1**
**Effort: ~2 giờ**

**Mục tiêu:** Calibrate xác suất output của model để "63% confidence" thực sự có ý nghĩa.

**File cần tạo/sửa:**

1. **Tạo `scripts/calibrate_temperature.py`**:
   ```python
   # Pseudo-code
   # 1. Load best model checkpoint
   # 2. Chạy inference trên validation set → thu logits
   # 3. Tìm optimal temperature T bằng NLL minimization
   # 4. Lưu T vào checkpoint
   # 5. Vẽ Reliability Diagram trước/sau
   ```

2. **Sửa `src/deployment/predictor.py`**:
   - Thêm `self.temperature` parameter
   - `probs = softmax(logits / self.temperature)` thay vì `softmax(logits)`

**Output:**
- [ ] File `results/paper_figures/calibration_curve.png`
- [ ] Giá trị T tối ưu được ghi vào checkpoint
- [ ] Báo cáo ECE (Expected Calibration Error) trước/sau

---

#### Nhiệm vụ C3: Keyword Leakage Test 🟡
**Deadline: Tuần 2**
**Effort: ~3 giờ**

**Mục tiêu:** Kiểm tra xem model có "gian lận" bằng cách học từ khóa shortcut ("fake", "hoax") không.

**File cần tạo:** `scripts/test_keyword_leakage.py`

**Logic:**
1. Load test set graphs
2. Tìm tất cả node features chứa từ khóa: `["fake", "hoax", "source", "debunked", "misleading", "false"]`
3. Thay embedding các node đó = mean embedding (mask)
4. Chạy inference trên masked test set
5. So sánh accuracy/F1 trước và sau mask

**Tiêu chí:**
- Nếu `delta < 5 pp` → "Model không phụ thuộc shortcut keyword" ✅
- Nếu `delta >= 5 pp` → Cần báo Leader để điều tra thêm ⚠️

**Output:**
- [ ] File `results/keyword_leakage_report.json`
- [ ] Bảng so sánh: Accuracy gốc vs Accuracy sau mask

---

#### Nhiệm vụ C4: Missing Modality Inference Test 🟡
**Deadline: Tuần 2**
**Effort: ~3 giờ**

**Mục tiêu:** Test chính thức xem model hoạt động thế nào khi thiếu image hoặc thiếu comment.

**File cần tạo:** `scripts/test_missing_modality.py`

**3 kịch bản test:**

| Kịch bản | Mô tả | Cách làm |
|----------|-------|---------|
| Full | Đầy đủ text + image + graph | Chạy bình thường |
| No Image | Thiếu ảnh | Zero-out toàn bộ image features |
| No Comment | Thiếu bình luận | Chỉ giữ root node, xóa hết comment nodes + edges |

**Output:**
- [ ] File `results/missing_modality_report.json`
- [ ] Bar chart so sánh accuracy 3 kịch bản → `results/paper_figures/missing_modality_comparison.png`

---

## 🟡 GIAI ĐOẠN 2 — BỔ SUNG (Tuần 2-3)

### 👤 THÀNH VIÊN A — Tiếp tục cào + Enrich

#### Nhiệm vụ A4: Enrich comment tree cho data mới
**Deadline: Tuần 2**

```powershell
python src/utils/research_pipeline.py enrich --limit 100 --delay 1.0
```

**Deliverable:**
- [ ] File `data/reddit_enriched_data.jsonl` đã cập nhật
- [ ] File `data/reddit_enriched_binary.jsonl` đã refresh

---

#### Nhiệm vụ A5: Build graph cho data mới
**Deadline: Tuần 2**

```powershell
python src/utils/research_pipeline.py build-graphs --multimodal
```

**Deliverable:**
- [ ] Graphs mới trong `data/processed_graphs_multimodal/`
- [ ] Báo cáo: số graph mới được tạo

---

### 👤 THÀNH VIÊN C — Cải tiến nâng cao

#### Nhiệm vụ C5: Learned [MISSING] Embedding 🟢
**Deadline: Tuần 3 (nếu còn thời gian)**
**Effort: ~4 giờ**

**Mục tiêu:** Thay zero-padding khi thiếu ảnh bằng learned embedding.

**Sửa `src/training/train_multimodal_gnn.py`:**
```python
self.missing_image_emb = nn.Parameter(torch.randn(image_dim))

# Trong forward:
if image_feature.sum() == 0:
    image_feature = self.missing_image_emb
```

**Deliverable:**
- [ ] Model mới đã train với missing_image_emb
- [ ] So sánh accuracy trước/sau → ghi vào ablation

---

## 📅 TIMELINE TỔNG THỂ

```
Tuần 1:
├── Ngày 1-2: 🔴 Song song
│   ├── Leader:      TN5 + TN1 (Cascade + Ablation)
│   ├── Thành viên A: Cào 50 bài Reddit (A1)
│   ├── Thành viên B: Đọc papers cho bảng so sánh (B3)
│   └── Thành viên C: UNCERTAIN zone + Temperature (C1, C2)
│
├── Ngày 3-4: 🔴 Song song
│   ├── Leader:      TN2 + TN3 (Error Analysis + Robustness)
│   ├── Thành viên A: Gán nhãn 50 bài (A2)
│   ├── Thành viên B: Cross-check nhãn (B1)
│   └── Thành viên C: Keyword Leakage Test (C3)
│
├── Ngày 5: 🔴 Tổng hợp
│   ├── Leader:      TN4 Domain Shift (dùng data A cào)
│   └── Tất cả:      Review kết quả tuần 1

Tuần 2:
├── Leader:      Cải tiến model (Explainability, Temporal Encoding)
├── Thành viên A: Cào thêm 100+ bài (A3) + Enrich (A4) + Build graph (A5)
├── Thành viên B: Gán nhãn batch mới (B2) + Bảng so sánh (B3)
└── Thành viên C: Missing Modality Test (C4)

Tuần 3:
├── Leader:      Finalize experiments, review tất cả kết quả
├── Thành viên B: Làm slide + báo cáo (B4)
├── Thành viên C: Learned [MISSING] Embedding (C5, nếu kịp)
└── Tất cả:      Review báo cáo + Chuẩn bị bảo vệ
```

---

## 📌 QUY TẮC LÀM VIỆC CHUNG

### 1. Quy tắc commit code
```
git commit -m "[MEMBER_X] <mô tả ngắn>"
```
Ví dụ:
- `[MEMBER_A] Crawl 50 OOD test posts`
- `[MEMBER_C] Add UNCERTAIN zone to predictor`

### 2. Quy tắc đặt tên file output
- Data: `data/<tên_mô_tả>.jsonl`
- Figure: `results/paper_figures/<tên_thí_nghiệm>.png`
- Report: `results/<tên_báo_cáo>.json` hoặc `.md`

### 3. Communication
- Báo cáo tiến độ: **hàng ngày** trên group chat
- Format báo cáo:
  ```
  [NGÀY] [TÊN]
  ✅ Đã hoàn thành: ...
  🔄 Đang làm: ...
  ❌ Blocked: ... (nếu có)
  ```

### 4. Khi gặp lỗi
- **Merge bị trùng:** Dùng `research_pipeline.py merge-labels`, KHÔNG append tay
- **Train fail:** Kiểm tra đã chạy `merge-labels` + `enrich` chưa
- **GNN fail:** Chạy lại `build-graphs`
- **Không biết cách làm:** Hỏi Leader TRƯỚC khi tự xử lý

---

## ✅ CHECKLIST TỔNG — THEO DÕI TIẾN ĐỘ

### Giai đoạn 1 (Tuần 1) 🔴
- [ ] **A1** — Cào 50 bài Reddit OOD
- [ ] **A2** — Gán nhãn 50 bài
- [ ] **B1** — Cross-check nhãn
- [ ] **C1** — UNCERTAIN zone
- [ ] **C2** — Temperature Scaling
- [ ] **Leader** — TN1-TN5 hoàn thành

### Giai đoạn 2 (Tuần 2) 🟡
- [ ] **A3** — Cào thêm 100+ bài
- [ ] **A4** — Enrich comment tree
- [ ] **A5** — Build graph
- [ ] **B2** — Gán nhãn batch mới
- [ ] **B3** — Cross-domain comparison table
- [ ] **C3** — Keyword Leakage Test
- [ ] **C4** — Missing Modality Test

### Giai đoạn 3 (Tuần 3) 🟢
- [ ] **B4** — Slide + Báo cáo
- [ ] **C5** — Learned [MISSING] Embedding (optional)
- [ ] **ALL** — Review tổng thể
- [ ] **ALL** — Chuẩn bị bảo vệ

---

## 📎 TÀI LIỆU THAM KHẢO

| Tài liệu | Đường dẫn | Ai cần đọc |
|-----------|-----------|------------|
| Hướng dẫn gán nhãn | `docs/LABELING_GUIDE.md` | A, B |
| Pipeline hướng dẫn | `docs/TEAM_RESEARCH_PIPELINE.md` | A, C |
| Kiến trúc hệ thống | `docs/ARCHITECTURE_EXPLANATION.md` | C |
| Kế hoạch thí nghiệm | `docs/WINNING_EXPERIMENT_PLAN.md` | Leader |
| Kế hoạch cải tiến | `docs/IMPROVEMENT_PLAN.md` | C, Leader |
| Q&A bảo vệ (70 câu) | `docs/DEFENSE_QA_70_ANSWERS.md` | B (để làm slide) |

---

> [!IMPORTANT]
> **Mọi output (data, figure, report) phải được push lên Git NGAY khi hoàn thành.**
> Leader sẽ review và integrate vào pipeline chính.
