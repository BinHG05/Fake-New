# TÀI LIỆU CUNG CẤP NGỮ CẢNH CHO NOTEBOOKLM: PHÁT HIỆN TIN GIẢ ĐA PHƯƠNG THỨC BẰNG CASCADE GNN

Tài liệu này được soạn thảo nhằm cung cấp thông tin nền tảng, quy trình thực hiện, và danh sách tài liệu tham khảo chi tiết của dự án "Phát hiện Tin giả Đa phương thức trên Mạng xã hội sử dụng Mạng Nơ-ron Đồ thị (Cascade GNN)". NotebookLM hãy sử dụng nội dung này như "Ground Truth" để hiểu sâu về dự án và viết Báo cáo Nghiên cứu Khoa học (NCKH) chuyên sâu, có trích dẫn khoa học chuẩn xác.

---

## PHẦN 1: TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

**1. Mục tiêu Nghiên cứu:**
Xây dựng một hệ thống phát hiện tin giả (Fake News Detection) tự động và hiệu quả trên mạng xã hội (cụ thể là Reddit) bằng cách kết hợp 3 luồng thông tin (đa phương thức):
- **Văn bản (Text):** Nội dung bài viết và bình luận.
- **Hình ảnh (Image):** Ảnh đính kèm trong bài viết gốc.
- **Cấu trúc lan truyền (Propagation Structure):** Chuỗi bình luận (comment tree) thể hiện cách cộng đồng mạng phản ứng lại với bài viết.

**2. Kiến trúc Mô hình Đề xuất (Multimodal Cascade GNN):**
Mô hình khắc phục hạn chế của các phương pháp cũ (chỉ dùng văn bản hoặc chỉ phân tích bài gốc) bằng cách khai thác tính lan truyền của tin giả.
- **Mã hóa Văn bản (Text Encoder):** Sử dụng XLM-RoBERTa (base) để trích xuất đặc trưng văn bản đa ngữ (768 chiều) cho mỗi bài viết và bình luận (tức là mỗi node trên đồ thị).
- **Mã hóa Hình ảnh (Image Encoder):** Sử dụng ResNet50 (pre-trained trên ImageNet) để trích xuất đặc trưng hình ảnh (512 chiều) từ ảnh của bài viết gốc.
- **Kết hợp Đa phương thức (Fusion Module):** Sử dụng cơ chế Cross-Attention hoặc Gated Fusion để trộn thông tin văn bản và hình ảnh tại node gốc. Các node bình luận (không có ảnh) sẽ được padding bằng vector 0.
- **Học Biểu diễn Đồ thị (Graph Neural Network):** Sử dụng GraphSAGE (2 layers) để tổng hợp thông tin từ cấu trúc chuỗi bình luận. Sau đó áp dụng Global Mean Pooling để tạo ra vector biểu diễn toàn cục cho cả đồ thị (Graph Embedding).
- **Phân loại (Classifier):** MLP phân loại tin giả theo nhị phân (REAL/FAKE) hoặc 6 mức độ tinh chỉnh (6-Class: True, Mostly True, Half True, Barely True, False, Pants on Fire).

---

## PHẦN 2: QUY TRÌNH THỰC HIỆN, TEST VÀ TRAIN (WORKFLOW)

Dự án sử dụng một Pipeline tự động, chuẩn hóa từ khâu thu thập đến huấn luyện, được vận hành qua script `research_pipeline.py`. Quy trình NCKH bao gồm 6 bước chính:

### Bước 1: Thu thập dữ liệu (Data Crawling & Scraping)
- Sử dụng Reddit API (PRAW) để lấy các bài viết mới từ mạng xã hội, đặc biệt ưu tiên các bài viết có đính kèm hình ảnh (`--images-only`) để làm phong phú dữ liệu đa phương thức.
- Dữ liệu thô (raw data) chứa metadata, ảnh, văn bản được lưu lại thành các batch.

### Bước 2: Tiền xử lý và Gán nhãn tự động (Pre-processing & Auto-labeling)
- Do việc gán nhãn thủ công tốn thời gian, nhóm nghiên cứu áp dụng LLM (Mô hình ngôn ngữ lớn) để gán nhãn tự động bước đầu (Auto-labeling).
- Quá trình này tạo ra các file batch định dạng JSON tương thích với công cụ gán nhãn Label Studio.

### Bước 3: Đánh giá của con người (Human Review & Labeling)
- Dữ liệu được đưa lên **Label Studio**. Các chuyên gia/thành viên trong nhóm tiến hành thẩm định lại các nhãn do LLM tạo ra nhằm đảm bảo độ tin cậy tuyệt đối của tập dữ liệu huấn luyện (Ground Truth).
- Dữ liệu sau đó được export và gộp (merge) vào một file "Master Labeled Data", tự động quy đổi nhãn 6-class sang nhị phân (Binary: REAL/FAKE) để tối ưu cho phân loại.

### Bước 4: Làm giàu dữ liệu cấu trúc (Tree Enrichment)
- Tin giả thường có cấu trúc phản ứng của cộng đồng rất đặc trưng (người dùng hay nghi ngờ, tranh cãi).
- Hệ thống gọi API Reddit để lấy toàn bộ "Comment Tree" (cây bình luận) của các bài viết đã được gán nhãn. Mỗi bài viết trở thành gốc của một đồ thị, các bình luận là các node con, quan hệ reply-to là các cạnh.

### Bước 5: Xây dựng Đồ thị Đa phương thức (Graph Construction)
- Đối với mỗi bài viết, hệ thống khởi tạo một đồ thị có hướng (Cascade Graph).
- **Trích xuất đặc trưng Node:** Chạy XLM-RoBERTa cho text và ResNet50 cho ảnh. Ghép nối lại thành vector 1280 chiều cho mỗi node (node không có ảnh sẽ zero-padded phần image).
- Lưu đồ thị dưới dạng tensor (PyTorch Geometric `Data` object).

### Bước 6: Huấn luyện và Đánh giá (Model Training & Testing)
- **Train:** Huấn luyện mô hình sử dụng PyTorch và PyTorch Geometric. Hàm Loss là Cross-Entropy có đánh trọng số (Weighted Loss) để xử lý việc mất cân bằng dữ liệu (Class Imbalance). Tối ưu hóa bằng Adam Optimizer, kết hợp Early Stopping.
- **Baseline Models:** Huấn luyện song song các mô hình cơ sở để so sánh:
  - *Text-Only:* Chỉ dùng nội dung chữ.
  - *Image-Only:* Chỉ dùng ảnh.
  - *Late Fusion:* Nối text và image nhưng KHÔNG dùng cấu trúc đồ thị.
  - *Graph GCN / GAT:* Dùng GNN nhưng chỉ với text.
- **Test / Evaluation:** Sử dụng tập Hold-out (Validation/Test set) để đánh giá khả năng tổng quát hóa. Đánh giá dựa trên các Metrics khoa học: **Binary Accuracy, Macro-F1**. Thực hiện *Ablation Study* (bỏ từng phần text/image/graph) để chứng minh đóng góp của từng thành phần.
- **Thí nghiệm Multi-Seed:** Huấn luyện trên nhiều random seeds khác nhau để đảm bảo độ ổn định (robustness) và tính tin cậy thống kê của mô hình.

---

## PHẦN 3: DANH SÁCH TÀI LIỆU THAM KHẢO (REFERENCE PAPERS)

NotebookLM có thể sử dụng các bài báo này làm Reference (Nguồn tham khảo) để viết phần Giới thiệu (Introduction), Công trình liên quan (Related Work) và Giải thích kiến trúc.

**1. Về Phát hiện Tin giả Đa phương thức (Multimodal Fake News Detection):**
*   **Singhal, S., et al. (2019). "SpotFake: A Multi-modal Framework for Fake News Detection."** *IEEE BigMM.*
    *   *Mô tả:* Tiên phong trong việc sử dụng BERT cho văn bản và VGG19 cho hình ảnh để phát hiện tin giả. Chứng minh rằng việc kết hợp đa phương thức hiệu quả hơn đơn phương thức.
*   **Khattar, D., et al. (2019). "MVAE: Multimodal Variational Autoencoder for Fake News Detection."** *The World Wide Web Conference (WWW).*
    *   *Mô tả:* Sử dụng VAE để học một không gian biểu diễn chung cho hình ảnh và văn bản. Rất hữu ích khi lập luận về tầm quan trọng của việc nén và trích xuất đặc trưng chung.
*   **Nakamura, K., et al. (2020). "r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection."** *LREC.*
    *   *Mô tả:* Bài báo công bố tập dữ liệu Fakeddit, sử dụng phân loại 6 mức độ (2-way, 3-way, 6-way). Cung cấp cơ sở khoa học cho quyết định phân loại theo 6 mức độ tin cậy của nhóm nghiên cứu.

**2. Về Mạng Nơ-ron Đồ thị cho Lan truyền Thông tin (Graph Neural Networks for Information Cascades):**
*   **Bian, T., et al. (2020). "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks."** *AAAI.*
    *   *Mô tả:* Công trình kinh điển (BiGCN) sử dụng GCN để mô hình hóa đồ thị lan truyền từ trên xuống (top-down) và từ dưới lên (bottom-up) của tin đồn trên Twitter. Củng cố lý thuyết rằng phản ứng của người dùng (bình luận) mang tín hiệu mạnh để phát hiện tin giả.
*   **Lu, Y.J., & Li, C.T. (2020). "GCAN: Graph-aware Co-Attention Networks for Detecting Fake News on Social Media."** *ACL.*
    *   *Mô tả:* Sử dụng mạng đồ thị kết hợp Attention để phân tích cách người dùng chia sẻ tin tức theo thời gian. Ủng hộ việc áp dụng Attention trên graph.
*   **Hamilton, W.L., et al. (2017). "Inductive Representation Learning on Large Graphs."** *NeurIPS.*
    *   *Mô tả:* Bài báo giới thiệu kiến trúc **GraphSAGE**, lý thuyết nền tảng cho layer GNN được nhóm sử dụng. GraphSAGE giúp hệ thống có khả năng tổng quát hóa trên các đồ thị (bài viết) chưa từng thấy trong quá trình huấn luyện.

**3. Về Kiến trúc Trích xuất Đặc trưng (Feature Extractors):**
*   **Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision."** *ICML.*
    *   *Mô tả:* Bài báo giới thiệu kiến trúc **CLIP (Contrastive Language-Image Pre-training)**. Đây là linh hồn của phiên bản V3 trong project, cho phép mô hình hiểu được sự tương quan giữa hình ảnh và văn bản thông qua không gian embedding chung. Việc sử dụng CLIP giúp phát hiện các trường hợp "râu ông nọ chắp cằm bà kia" (hình ảnh và nội dung không khớp nhau) - một dấu hiệu đặc trưng của tin giả.
*   **Conneau, A., et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale."** *ACL.*
    *   *Mô tả:* Cung cấp nền tảng khoa học cho **XLM-RoBERTa**. Việc chọn XLM-RoBERTa giúp mô hình xử lý văn bản đa ngôn ngữ, đặc biệt quan trọng cho việc chuyển đổi tri thức từ tiếng Anh sang tiếng Việt (Cross-lingual Transfer).
*   **He, K., et al. (2016). "Deep Residual Learning for Image Recognition."** *CVPR.*
    *   *Mô tả:* Nền tảng của **ResNet50**. Dù trong V3 đã chuyển sang dùng CLIP-ViT làm backbone chính, nhưng tư duy về Residual Learning vẫn là nền tảng cho các bộ trích xuất đặc trưng hình ảnh hiện đại.
*   **Vaswani, A., et al. (2017). "Attention Is All You Need."** *NeurIPS.*
    *   *Mô tả:* Khởi xướng cơ chế Attention. Nhóm nghiên cứu dựa vào đây để thiết kế "Cross-Attention Fusion Module", giúp text embedding "tập trung" vào các đặc trưng quan trọng của image embedding.

---
**Gợi ý cho NotebookLM khi viết báo cáo:**
- Nhấn mạnh vào **khoảng trống nghiên cứu (Research Gap):** Hầu hết các phương pháp hiện nay chỉ là "Multimodal" (Text + Image) hoặc "Graph-based" (Text + Graph). Dự án này kết hợp cả ba: **Text + Image + Graph**.
- Phân tích sâu bước Ablation Study trong quy trình (Bước 6): Khi tắt mô đun Hình ảnh hoặc Đồ thị, độ chính xác (Accuracy) giảm mạnh, điều này chứng minh tính đúng đắn của giả thuyết nghiên cứu.
