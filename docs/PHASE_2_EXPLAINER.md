# 🌐 PHASE 2: CRAWLING DỮ LIỆU "LAN TRUYỀN" (CASCADE)

Tài liệu này giải thích **Giai đoạn 2** của dự án cho team dev và người làm dữ liệu.

## 1. Mục tiêu: "Đừng chỉ đọc tin, hãy xem người ta nói gì về nó"

Ở Phase 1, chúng ta chỉ lấy **Nội dung bài viết (Post)**. Nhưng Fake News thường bị lộ tẩy qua **Comment**.

> **Ví dụ:**
> *   **Tin giả:** "Ăn tỏi chữa ung thư"
> *   **Comment 1:** "Xạo quá, tỏi chỉ là gia vị thôi."
> *   **Comment 2:** "Bác sĩ đã bác bỏ tin này rồi."
> *   **Comment 3:** [Dẫn link bài báo đính chính]

=> **Mục tiêu Phase 2:** Thu thập toàn bộ **Cây bình luận (Cascade Tree)** của bài viết để AI phân tích xem cộng đồng phản ứng thế nào.

---

## 2. Code: Chúng ta code cái gì?

Chúng ta cần code 2 file chính trong giai đoạn này:

### A. Người đi lấy tin: `reddit_crawler.py`
*   **Nhiệm vụ:** Không chỉ lấy bài post, mà phải "chui" vào link bài viết, lấy hết các comment cha, comment con.
*   **Logic:**
    1.  Lấy ID bài viết.
    2.  Gửi request lấy Comment.
    3.  Lưu lại ai bình luận? Bình luận lúc nào? Trả lời cho ai?
*   **Kết quả:** File JSONL, mỗi dòng không chỉ có `post` mà có thêm `cascade` (danh sách comment).

```json
/* Output mẫu */
{
  "id": "post_123",
  "text": "Tin nóng: ABC...",
  "cascade": [
    {"user": "UserA", "text": "Fake news!", "reply_to": "post_123"},
    {"user": "UserB", "text": "Đúng rồi", "reply_to": "UserA"}
  ]
}
```

### B. Kiến trúc sư: `cascade_graph_builder.py`
*   **Nhiệm vụ:** Máy tính không hiểu JSON, nó cần **Đồ thị (Graph)** để chạy thuật toán GNN. File này chuyển JSON comment thành các cục (Node) và dây nối (Edge).
*   **Logic:**
    1.  **Node:** Mỗi comment là 1 chấm tròn. Chứa vector văn bản (dùng BERT/XLM-R để hiểu nghĩa).
    2.  **Edge:** Nếu User B trả lời User A -> Vẽ mũi tên từ A sang B.
*   **Kết quả:** Đối tượng `Data` của PyTorch Geometric (sẵn sàng để train mô hình).

---

## 3. Kết quả (Deliverable) của Phase 2

Kết thúc giai đoạn này, chúng ta phải có được:

1.  **Bộ dữ liệu mở rộng:**
    *   Khoảng 5,000 - 10,000 bài viết.
    *   Mỗi bài đi kèm trung bình 20-50 comment.
    *   Tổng cộng hàng trăm nghìn node dữ liệu xã hội.

2.  **Công cụ (Code):**
    *   Crawler chạy ổn định, không bị Reddit chặn.
    *   Class `CascadeGraphBuilder` hoạt động tốt, biến comment thành graph.

## 4. Tại sao cái này quan trọng?

Nếu không có Phase 2, dự án của chúng ta chỉ là **Phân loại văn bản bình thường** (như lọc spam email).
Có Phase 2, dự án trở thành **Phân tích mạng xã hội (Social Network Analysis)** -> Đây là điểm ăn tiền (Novelty) của đề tài NCKH này.
