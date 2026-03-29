# Kế Hoạch Chuyển Đổi Từ 6 Class Sang Binary

## 1. Mục tiêu

Chuyển bài toán từ phân loại 6 mức độ thật giả:

- `TRUE`
- `MOSTLY_TRUE`
- `HALF_TRUE`
- `BARELY_TRUE`
- `FALSE`
- `PANTS_ON_FIRE`

sang bài toán nhị phân:

- `REAL`
- `FAKE`

Mục tiêu của việc chuyển đổi là:

- đưa bài toán về đúng bản chất hơn với use case "fake news detection"
- giảm độ khó của tác vụ phân loại
- tăng độ ổn định của mô hình
- cải thiện accuracy/F1 để phù hợp hơn với yêu cầu nghiên cứu

## 2. Nguyên tắc nghiên cứu

Việc chuyển đổi này phải được trình bày như một thay đổi có cơ sở học thuật, không phải chỉ để làm đẹp kết quả.

Nguyên tắc áp dụng:

- giữ nguyên nhãn gốc 6 class trong dữ liệu
- tạo thêm nhãn mới `label_binary`
- báo cáo rõ rằng dữ liệu gốc là 6 class, nhưng bài toán chính được chuẩn hóa thành binary classification
- nếu cần, vẫn giữ 6-class như thí nghiệm phụ hoặc phần mở rộng

## 3. Quy tắc ánh xạ nhãn

Ánh xạ đề xuất:

- `TRUE` -> `REAL`
- `MOSTLY_TRUE` -> `REAL`
- `HALF_TRUE` -> `REAL`
- `BARELY_TRUE` -> `FAKE`
- `FALSE` -> `FAKE`
- `PANTS_ON_FIRE` -> `FAKE`

Lưu ý:

- `HALF_TRUE` là lớp mơ hồ nhất, cần nêu rõ trong báo cáo vì sao nhóm xếp vào `REAL`
- nếu kết quả sau này chưa tốt, có thể mở thêm thí nghiệm phụ với biến thể xử lý riêng cho `HALF_TRUE`

## 4. Các giai đoạn thực hiện

### Giai đoạn 1: Chuẩn hóa định nghĩa bài toán

Việc cần làm:

- thống nhất trong nhóm rằng bài toán chính là `binary fake news detection`
- cập nhật mô tả đề tài trong báo cáo, slide, docs
- xác định metric chính sẽ là:
  - `Accuracy`
  - `F1`
  - `Precision`
  - `Recall`
  - `Confusion Matrix`

Output:

- định nghĩa bài toán mới rõ ràng trong tài liệu nghiên cứu

### Giai đoạn 2: Chuyển đổi dữ liệu

Việc cần làm:

- giữ lại trường `label` gốc
- sinh thêm trường `label_binary`
- cập nhật các file dữ liệu trung tâm:
  - `data/03_clean/Fakeddit/labeled_master.jsonl`
  - `data/reddit_enriched_data.jsonl`
  - các file split train/val/test liên quan
- kiểm tra phân bố lớp sau khi chuyển đổi

Mục tiêu của bước này:

- không làm mất dữ liệu cũ
- cho phép train lại theo binary mà vẫn truy vết được nguồn gốc 6-class

Output:

- dataset binary-ready
- thống kê phân bố `REAL/FAKE`

### Giai đoạn 3: Cập nhật pipeline data loading

Việc cần làm:

- sửa dataset loader để đọc `label_binary`
- giữ khả năng fallback nếu file cũ chưa có trường mới
- đảm bảo tất cả split train/val/test vẫn hoạt động đúng

Các vị trí cần xem:

- `src/data/text_image_dataset.py`
- các loader trong script train GNN
- các bước merge/convert từ Label Studio

Output:

- data loader nhất quán cho binary classification

### Giai đoạn 4: Cập nhật training scripts

Việc cần làm:

- chuyển toàn bộ baseline models sang train theo 2 class
- chuyển GNN và multimodal GNN sang train theo 2 class
- chuẩn hóa output metric giữa các script
- cập nhật class weights cho bài toán binary

Các file chính:

- `src/training/train_baseline.py`
- `src/training/train_gnn.py`
- `src/training/train_multimodal_gnn.py`

Output:

- tất cả mô hình dùng chung một target nhị phân

### Giai đoạn 5: Cập nhật experiments và dashboard

Việc cần làm:

- sửa phần parse kết quả để dùng metric binary làm metric chính
- sửa bảng so sánh model
- phân biệt rõ kết quả cũ 6-class và kết quả mới binary
- cập nhật biểu đồ, report, run history nếu cần

Các vị trí chính:

- `src/experiments/run_paper_experiments.py`
- `src/experiments/visualize_results.py`
- `demo-website/app.py`
- `demo-website/script.js`

Output:

- hệ thống hiển thị đúng kết quả binary

### Giai đoạn 6: Train lại và đánh giá

Việc cần làm:

- train lại toàn bộ mô hình:
  - Text-only
  - Image-only
  - Fusion
  - Graph GNN
  - Multimodal GNN
- chạy nhiều seed nếu có thời gian
- so sánh model trên cùng một bài toán binary

Metric cần báo cáo:

- Accuracy
- F1-score
- Precision
- Recall
- Confusion Matrix

Output:

- bảng kết quả binary đầy đủ
- mô hình tốt nhất cho báo cáo

### Giai đoạn 7: Viết lại phần giải thích trong báo cáo

Việc cần làm:

- giải thích vì sao 6-class khó hơn binary
- giải thích vì sao binary phù hợp hơn với mục tiêu phát hiện fake news
- mô tả mapping nhãn
- nêu rõ đây là quyết định thiết kế bài toán, không phải thay đổi tùy tiện

Nội dung nên có:

- lý do học thuật
- lý do thực tiễn
- ảnh hưởng tới metric
- giới hạn của hướng binary

Output:

- phần methodology và experimental setup chặt chẽ hơn

## 5. Rủi ro cần lưu ý

- metric tăng do bài toán dễ hơn, nên phải ghi rõ trong báo cáo
- không được so sánh trực tiếp kết quả 6-class cũ với kết quả binary mới như thể cùng một task
- nếu dữ liệu bị lệch lớp mạnh sau khi gộp, cần xử lý bằng class weight hoặc resampling
- cần kiểm tra kỹ `HALF_TRUE` vì đây là lớp gây tranh cãi nhất

## 6. Chiến lược khuyến nghị

Khuyến nghị thực hiện theo thứ tự sau:

1. Không gán nhãn lại từ đầu.
2. Tạo thêm `label_binary` từ dữ liệu 6-class hiện có.
3. Chuyển toàn bộ pipeline train/eval sang binary.
4. Train lại toàn bộ mô hình.
5. Dùng binary làm kết quả chính trong nghiên cứu.
6. Giữ 6-class làm baseline phụ hoặc phần thảo luận mở rộng.

## 7. Deliverables mong muốn

Sau khi hoàn thành chuyển đổi, project cần có:

- dataset có cả `label` và `label_binary`
- training scripts binary-ready
- bảng kết quả binary cho tất cả model
- dashboard và experiment scripts đồng bộ
- tài liệu giải thích rõ quyết định chuyển đổi bài toán

## 8. Checklist thực thi

- [ ] Thống nhất lại mục tiêu nghiên cứu thành binary detection
- [ ] Tạo trường `label_binary` trong dữ liệu
- [ ] Cập nhật loader dữ liệu
- [ ] Cập nhật baseline training
- [ ] Cập nhật GNN training
- [ ] Cập nhật multimodal training
- [ ] Cập nhật experiment runner
- [ ] Cập nhật dashboard
- [ ] Train lại toàn bộ model
- [ ] Tổng hợp kết quả và viết lại phần methodology/report

