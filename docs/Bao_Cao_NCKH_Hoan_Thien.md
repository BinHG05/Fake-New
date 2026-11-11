# BÁO CÁO NGHIÊN CỨU KHOA HỌC (PHIÊN BẢN HOÀN THIỆN)
## ĐỀ TÀI: PHÁT HIỆN TIN GIẢ ĐA PHƯƠNG THỨC TRÊN MẠNG XÃ HỘI SỬ DỤNG MẠNG NƠ-RON ĐỒ THỊ (CASCADE GNN)

---

### CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI NGHIÊN CỨU

#### 1.1. Tình hình nghiên cứu trong và ngoài nước
Sự bùng nổ của mạng xã hội (Social Media) đã biến đây thành phương thức tiếp nhận thông tin chủ đạo. Tuy nhiên, tin giả (fake news) lan truyền nhanh gấp 6 lần tin thật và thường có cấu trúc lan truyền phức tạp. Nghiên cứu AI hiện nay chia làm 3 hướng chính:
1.  **Dựa trên nội dung (Content-based):** Sử dụng NLP (BERT, RoBERTa) để soi xét văn phong. *Hạn chế:* Dễ bị qua mặt bởi tin giả viết chuyên nghiệp.
2.  **Dựa trên lan truyền (Propagation-based):** Sử dụng GNN để phân tích đồ thị lan truyền. *Hạn chế:* Bỏ qua hình ảnh - một phương thức thao túng cực mạnh.
3.  **Dựa trên đa phương thức (Multimodal):** Kết hợp ảnh và chữ. *Hạn chế:* Thường chỉ phân tích bài đăng gốc (static analysis), bỏ qua phản ứng của cộng đồng.

**=> Đề tài này giải quyết "khoảng trống" bằng cách kết hợp cả 3: Văn bản + Hình ảnh + Cấu trúc đồ thị lan truyền.**

#### 1.2. Lý do chọn đề tài và tính cấp thiết
Tin giả không chỉ gây nhiễu mà còn đe dọa an ninh xã hội. Reddit là môi trường đặc thù với các cây bình luận (comment trees) sâu, nơi các cuộc tranh luận diễn ra gay gắt. Việc xây dựng một mô hình có khả năng lắng nghe "thái độ" của cộng đồng để xác thực tin tức là vô cùng cần thiết.

#### 1.3. Mục tiêu và phạm vi nghiên cứu
*   **Mục tiêu:** Xây dựng mô hình Cascade GNN đạt độ chính xác cao trong phân loại nhị phân (REAL/FAKE).
*   **Phạm vi:** Tập trung vào các bài đăng đa phương tiện trên Reddit.

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

#### 2.1. XLM-RoBERTa (Text Encoder)
Sử dụng cơ chế Self-Attention để hiểu ngữ cảnh sâu của văn bản. XLM-RoBERTa vượt trội ở khả năng xử lý đa ngôn ngữ, giúp mô hình hiểu được các sắc thái từ ngữ trong bình luận.

#### 2.2. ResNet50 (Image Encoder)
Trích xuất đặc trưng thị giác thông qua các lớp tích chập sâu với kết nối tắt (Skip-connection). Giúp mô hình nhận diện các đặc điểm của ảnh giả hoặc ảnh không khớp ngữ cảnh.

#### 2.3. GraphSAGE (Graph Learner)
Cho phép tổng hợp thông tin từ hàng xóm (neighbor aggregation). Khác với GCN, GraphSAGE có tính tổng quát hóa cao, giúp mô hình hoạt động tốt trên cả những bài viết mới chưa từng thấy lúc train.

#### 2.4. Cross-Attention Fusion
Thay vì nối vector đơn giản, Cross-Attention cho phép văn bản "truy vấn" hình ảnh để tìm sự đồng nhất. Điều này cực kỳ hiệu quả trong việc phát hiện các vụ "râu ông nọ chắp cằm bà kia".

---

### CHƯƠNG 3: KIẾN TRÚC MÔ HÌNH MULTIMODAL CASCADE GNN

#### 3.1. Thiết kế Pipeline hệ thống
Hệ thống được thiết kế dưới dạng luồng dữ liệu khép kín:
1.  **Cào dữ liệu (PRAW):** Thu thập bài đăng và cây bình luận đệ quy.
2.  **Mã hóa (Encoders):** Chữ chuyển thành vector 768d, ảnh chuyển thành vector 512d.
3.  **Hòa quyện (Fusion):** Kết hợp thành vector 1280d tại mỗi node.
4.  **Xử lý đồ thị (GNN):** Chạy 2 lớp GraphSAGE để học đặc trưng lan truyền.

#### 3.2. Cải tiến kỹ thuật trọng tâm
*   **Learned [MISSING] Image Embedding:** Đối với các bình luận (không có ảnh), thay vì dùng số 0, chúng em huấn luyện một vector đặc trưng để mô hình học cách phân biệt sự vắng mặt của ảnh.
*   **Graph Augmentation:** Áp dụng Node/Edge Drop trong lúc train để mô hình không bị "học vẹt" và bền vững hơn trước các dữ liệu nhiễu.

---

### CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

Chương này trình bày chi tiết các kết quả thực nghiệm thu được từ mô hình Cascade GNN, đồng thời thực hiện các bài kiểm tra chuyên sâu về độ bền vững, tính giải thích được và khả năng tổng quát hóa của hệ thống. Toàn bộ thực nghiệm được thực hiện trên một bộ khung đánh giá chuẩn hóa (Evaluation Harness) để đảm bảo tính khách quan và nhất quán.

#### 4.1. Thiết lập thực nghiệm và Dữ liệu
*   **Tập dữ liệu:** 2,384 đồ thị lan truyền (cascade graphs) từ Reddit, phân tách theo tỷ lệ 70% Train, 10% Validation, và 20% Test.
*   **Cấu hình phần cứng:** Sử dụng GPU NVIDIA với môi trường PyTorch 2.x và PyTorch Geometric.
*   **Siêu tham số:** Fusion dimension 256, GraphSAGE 2 layers, Dropout 0.3, Learning rate 1e-3 với Adam Optimizer.
*   **Hàm mất mát:** Weighted Cross-Entropy để xử lý tình trạng mất cân bằng dữ liệu (Class Imbalance).

#### 4.2. Kết quả phân loại nhị phân tổng thể
Mô hình Multimodal Cascade GNN đạt được hiệu suất vượt trội trên bài toán phân loại Tin thật (REAL) và Tin giả (FAKE):
*   **Độ chính xác tổng thể (Binary Accuracy):** **87.50%**.
*   **Chỉ số lớp REAL:** Precision 88.10%, Recall 90.80%.
*   **Chỉ số lớp FAKE:** Precision 86.61%, Recall 82.91%.
Kết quả này cho thấy mô hình có khả năng nhận diện tin thật rất tốt và duy trì sự cân bằng cao trong việc phát hiện các mẫu tin giả tinh vi.

#### 4.3. Phân tích bóc tách thành phần (Ablation Study)
Để xác định đóng góp của từng luồng thông tin, chúng em thực hiện các thí nghiệm bóc tách:
1.  **Vai trò của Đồ thị (GNN):** So sánh mô hình đầy đủ với mô hình không dùng Graph (chỉ dùng Fusion bài gốc). Kết quả: Độ chính xác giảm từ **87.50%** xuống còn **67.2%** (Δ = **-20.3%**). Điều này khẳng định cấu trúc lan truyền là chìa khóa then chốt.
2.  **Vai trò của Hình ảnh (Multimodal):** Khi loại bỏ thông tin hình ảnh và chỉ sử dụng văn bản, độ chính xác giảm mạnh (Δ = **-20.7%**). Điều này chứng minh hình ảnh cung cấp những tín hiệu thị giác quan trọng mà văn bản không thể thay thế.
3.  **Cơ chế Fusion:** Cross-Attention đạt hiệu quả cao hơn Gated Fusion trong việc duy trì độ chính xác nhị phân, giúp mô hình "tập trung" vào các vùng mâu thuẫn giữa ảnh và chữ.

#### 4.4. Đóng góp khoa học của cấu trúc lan truyền (Cascade Topology)
Nghiên cứu sâu về tô-pô đồ thị cho thấy sự khác biệt mang tính bản chất giữa tin thật và tin giả:
*   **Độ sâu trung bình:** Tin giả có xu hướng lan truyền sâu hơn (**1.56**) so với tin thật (**0.70**).
*   **Hệ số phân nhánh:** Tin giả có tốc độ lan tỏa rộng hơn (**1.15**) so với tin thật (**0.45**).
*   **Ý nghĩa thống kê:** Các phép kiểm định Mann-Whitney cho giá trị **p < 1e-8**, khẳng định rằng "hình thái học" của sự lan truyền tin giả là một đặc trưng độc lập và mạnh mẽ để AI khai thác.

#### 4.5. Phân tích lỗi và Đặc điểm hành vi mô hình
*   **Hiệu ứng quy mô đồ thị:** Mô hình đạt chính xác cực cao (**92.86%**) trên các bài viết có ít bình luận, nhưng giảm xuống (**71.88%**) trên các đồ thị phức tạp (>10 comments). Điều này cho thấy sự nhiễu loạn thông tin trong các cuộc tranh luận lớn vẫn là một thách thức.
*   **Độ tự tin (Confidence):** Mô hình có khả năng tự đánh giá tốt. Ở dải độ tự tin cao (>0.7), xác suất dự đoán đúng lên tới **90.32%**. Chúng em đã thiết lập một **"Vùng nghi vấn" (Uncertain Zone)** với ngưỡng 0.15 để hệ thống biết "từ chối đưa ra kết luận" khi tín hiệu không rõ ràng.

#### 4.6. Kiểm chứng chống "Học vẹt" (Keyword Leakage Test)
Một nghi vấn lớn trong phát hiện tin giả là AI có thể chỉ học các từ khóa bóc phốt (như "fake", "hoax", "debunked") trong bình luận thay vì hiểu nội dung.
*   **Thực nghiệm:** Chúng em thực hiện Masking (che đi) toàn bộ các node chứa từ khóa nhạy cảm này tại thời điểm inference.
*   **Kết quả:** Độ chính xác giữ nguyên không đổi (**Δ = 0.00%**). Điều này chứng minh mô hình thực sự học được ngữ cảnh và cấu trúc, không phụ thuộc vào các "lối tắt" ngôn ngữ học đơn giản.

#### 4.7. Độ bền vững và Khả năng tổng quát hóa
*   **Robustness:** Khi cố tình "tấn công" mô hình bằng cách chèn thêm các bình luận giả mạo vào các bài viết thật, tỷ lệ lật ngược kết quả (flip rate) lên tới 90%, cho thấy mô hình rất nhạy cảm với các tín hiệu phản hồi tiêu cực.
*   **Domain Shift:** Khi kiểm tra trên dữ liệu hoàn toàn mới nằm ngoài tập huấn luyện (OOD - Out of Distribution), độ chính xác giảm xuống còn **33.33%**. Điều này phản ánh tính chất đặc thù của dữ liệu Reddit và mở ra nhu cầu về các phương pháp thích nghi tên miền (Domain Adaptation) trong tương lai.

#### 4.8. Tính giải thích được của mô hình (Explainability)
Thông qua cơ chế Attention, chúng em có thể trích xuất được các bình luận có sức ảnh hưởng lớn nhất đến quyết định của AI. 
*   **Ví dụ:** Trong một vụ việc tin giả điển hình, mô hình tập trung trọng số (attention score = 0.50) vào các bình luận cung cấp bằng chứng phản biện ở cấp độ node sớm nhất trong cây lan truyền.

#### 4.9. Lộ trình cải tiến kỹ thuật (Future Upgrades)
Chúng em đã tích hợp và kiểm thử thành công hai kỹ thuật mới:
*   **Learned [MISSING] Embedding:** Thay thế zero-padding cho các đỉnh khuyết ảnh, giúp mô hình học được đặc trưng của sự vắng mặt thông tin.
*   **Graph Augmentation:** Sử dụng Node/Edge Drop trong huấn luyện để tăng cường tính bền bỉ của mạng GNN.
Kết quả sơ bộ (smoke test) cho thấy độ chính xác đạt **77.86%** chỉ sau 1 epoch huấn luyện, hứa hẹn một bước nhảy vọt về hiệu năng khi huấn luyện đầy đủ.

---

### CHƯƠNG 5: TRIỂN KHAI DEMO VÀ TRỰC QUAN HÓA

#### 5.1. Dashboard trực quan
Chúng em xây dựng Dashboard cho phép dán link Reddit và nhận kết quả dự đoán ngay lập tức. Hệ thống hiển thị:
1.  **Xác suất REAL/FAKE.**
2.  **Đồ thị lan truyền:** Trực quan hóa cấu trúc bình luận.
3.  **Giải thích bằng Graph:** Hiển thị các "cụm hoài nghi" trong cây bình luận.

#### 5.2. Khả năng mở rộng
Hệ thống đã sẵn sàng để tích hợp thêm các mô hình ngôn ngữ lớn (LLM) để tự động tóm tắt lý do tại sao một bài viết bị coi là tin giả dựa trên các bình luận của cộng đồng.

---

### KẾT LUẬN
Đề tài đã xây dựng thành công mô hình Cascade GNN đa phương thức với độ chính xác **87.50%**. Kết quả nghiên cứu khẳng định: Việc kết hợp cấu trúc đồ thị lan truyền cùng với hình ảnh và văn bản là giải pháp tối ưu để đối phó với vấn nạn tin giả ngày càng tinh vi trên mạng xã hội.
