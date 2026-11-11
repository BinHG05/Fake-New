# 🧠 HỆ THỐNG PHÁT HIỆN TIN GIẢ ĐA PHƯƠNG THỨC (MULTIMODAL GNN FAKE NEWS DETECTION)
**Tài liệu giải thích kiến trúc hệ thống và các thuật toán cốt lõi**

Tài liệu này tổng hợp toàn bộ luồng hoạt động (Pipeline) và các công nghệ lõi được sử dụng trong hệ thống Demo phát hiện tin giả. Bạn có thể sử dụng nội dung này cho các slide thuyết trình hoặc báo cáo khoa học.

---

## 1. TỔNG QUAN LUỒNG HOẠT ĐỘNG (SYSTEM PIPELINE)

Hệ thống hoạt động theo nguyên lý mô phỏng lại cách một tin tức được lan truyền trên mạng xã hội, bao gồm 4 giai đoạn chính:

1. **Crawler (Thu thập dữ liệu)**: Khi người dùng nhập URL bài viết (Reddit), Crawler sẽ cào tiêu đề, văn bản gốc, hình ảnh và *toàn bộ cây bình luận (Comment Cascade)*.
2. **Feature Extraction (Trích xuất đặc trưng)**: Các AI Model phân tích ngữ nghĩa nội dung và hình ảnh của bài viết để biến đổi thành các vector toán học (Embeddings).
3. **Graph Construction (Cấu trúc hóa Đồ thị)**: Biểu diễn bài viết gốc và các bình luận dưới dạng Đồ thị lan truyền (Propagation Graph).
4. **Prediction (Dự đoán)**: Một mô hình Mạng nơ-ron Đồ thị (GNN) tiếp nhận toàn bộ dữ liệu trên để chấm điểm và phân loại **REAL** (Tin thật) hay **FAKE** (Tin giả).

---

## 2. CƠ SỞ NHẬN BIẾT TIN THẬT (THE BASIS OF "REAL" DETERMINATION)

Một câu hỏi rất hay mà hội đồng thường đặt ra: *"Dựa vào đâu máy tính biết đó là tin thật?"*
Mô hình AI của chúng ta không tự nhiên "biết" quy chuẩn của tin thật. Nó sử dụng phương pháp **Học có giám sát (Supervised Learning)** dựa trên hàng trăm ngàn mẫu dữ liệu từ bộ dataset khổng lồ (Fakeddit). Qua quá trình huấn luyện, AI đã nhận diện được **"DNA của Tin Thật"** thông qua 3 yếu tố hội tụ:

### A. Dấu hiệu Văn bản (Text Patterns qua XLM-RoBERTa)
*   **Tin giả (FAKE)** thường dùng từ ngữ có tính kích động, cảm xúc mạnh (giận dữ, sợ hãi), viết hoa vô tội vạ hoặc giật tít (Clickbait).
*   **Tin thật (REAL)** sử dụng văn phong báo chí: Khách quan, trung lập, rành mạch và chứa các thực thể có thật (Tên người, địa danh, con số cụ thể). Mạng nơ-ron XLM-R của chúng ta đã học được cách gom các từ vựng trung lập này vào "vùng tỷ lệ REAL cao" trong không gian toán học (Vector Space).

### B. Dấu hiệu Hình ảnh & Tính nhất quán (Image & Alignment qua CLIP)
*   **Sự nhất quán (Alignment)**: Đây là điểm "cốt tử". Trong tin thật, hình ảnh và tiêu đề mang tính chất bổ trợ trực tiếp cho nhau. Ví dụ: Tiêu đề nói "Tổng thống ký sắc lệnh", ảnh là cảnh ông ấy cầm bút ký $\rightarrow$ Model CLIP tính ra **Độ tương đồng Cosine (Cosine Similarity)** rất cao.
*   **Ngược lại**: Tin giả thường dùng ảnh cũ đem "râu ông nọ cắm cằm bà kia" hoặc ảnh meme, ảnh chế. CLIP ngay lập tức nhận ra sự bất đồng bộ $\rightarrow$ Similarity Score phản bội lại bài viết, hệ thống nghiêng về FAKE.

### C. Dấu hiệu Tâm lý Đám đông (Topology Đồ thị qua GNN)
*   Hãy tưởng tượng mâm cỗ có người nói thật, người nói dối, đám đông sẽ phản ứng thế nào?
*   **Tin thật**: Cấu trúc bình luận (Graph Topology) thường có dạng phẳng hoặc chia nhánh ngắn. Khán giả chủ yếu bổ sung thông tin (add context), đồng tình hoặc tranh luận dựa trên lập luận logic. GNN nhìn vào các cạnh (Edges) này và thấy sự "ổn định".
*   **Tin giả**: Gây ra cú sốc dư luận. Đồ thị bình luận thường xuất hiện các nhánh cãi vã cực sâu (Echo chambers). Các bình luận chứa từ khóa bóc phốt như "fake", "hoax", "source?". Mạng **Graph Attention Networks (GAT)** quét qua các node comment này, "chú ý" (Attention) vào các từ khóa bóc phốt và ngay lập tức phạt điểm uy tín của Bài gốc (Root Node).

$\rightarrow$ **Kết luận**: Bài viết chỉ được phán đoán là **REAL** khi và chỉ khi: Lời lẽ khách quan + Hình ảnh khớp 100% với nội dung + Cấu trúc lan truyền của đám đông không biểu hiện sự phẫn nộ/hoài nghi bất thường.

---

## 3. CÁC THUẬT TOÁN & MÔ HÌNH LÕI (CORE ALGORITHMS)

(Hệ thống của chúng ta đạt hiệu năng vượt trội là nhờ sự kết hợp (Fusion) của 3 nhánh thuật toán tiên tiến)

### 3.1. XLM-RoBERTa (Trích xuất Đa ngôn ngữ)
*   **Vai trò**: Phân tích văn bản (Title, Body, Comment).
*   **Cơ chế**: Dựa trên kiến trúc *Transformer*, sử dụng *Self-Attention* để hiểu ngữ cảnh của câu.
*   **Output**: Chuyển đổi mọi đoạn text thành vector `768` chiều.

### 3.2. CLIP (Contrastive Language-Image Pre-training)
*   **Vai trò**: Khởi tạo sự kết nối Đa phương thức (Hình ảnh + Văn bản).
*   **Output tổng CLIP**: $512 (Image) + 512 (Text) + 1 (Cosine Score) = 1025$ chiều.
> **=> TỔNG VECTOR CHO MỖI NODE:** $768 + 1025 = 1793$ chiều.

### 3.3. GNN - GAT (Graph Attention Networks)
*   **Vai trò**: Phân tích hành vi lan truyền của dư luận.
*   **Cơ chế**: Dùng **Attention Weights** đánh giá xem ai đang nói thật, ai đáng tin cậy.

---

## 4. CƠ CHẾ CROSS-ATTENTION FUSION (SỰ KẾT HỢP TINH TẾ)
Hệ thống không chỉ cộng gộp tĩnh (Concatenation) mà dùng **Cross-Attention**. Nó ép Điểm đồ thị (Graph), Điểm Text (XLM-R) và Điểm mâu thuẫn hình (CLIP) phải "vấn đáp" chéo nhau trước khi đưa ra quyết định cuối cùng thông qua một lóp classifier chung (Linear Neural Layer kết hợp Sigmoid/Softmax logic).

---

## 5. BÍ QUYẾT BẢO VỆ CHỐNG SẬP (ROBUSTNESS & FALLBACK ALGORITHM)
*   **Zero-padding**: Thay vì báo lỗi nếu mất ảnh, hệ thống tự động khởi tạo vector `[0, ..., 0]`. Trọng số vắng mặt được Cross-Attention dồn sang Text + Graph để duy trì Up-time 100%. Mất ảnh mô hình vẫn "sống"!
