# 🏆 ULTIMATE TESTING GUIDE: PHƯƠNG PHÁP KIỂM THỬ PIPELINE FAKE NEWS (A-Z)

Tài liệu này hướng dẫn cách kiểm thử toàn diện hệ thống để đảm bảo đạt điểm tối đa về độ ổn định, tính năng và khả năng xử lý lỗi.

---

## 📋 GIAI ĐOẠN 0: CHUẨN BỊ (PRE-FLIGHT CHECK)

Trước khi bắt đầu, hãy đảm bảo các yếu tố sau đã sẵn sàng:

1.  **Môi trường**: Luôn sử dụng Conda `multimodal_gnn`.
    ```powershell
    conda activate multimodal_gnn
    ```
2.  **Checkpoints**: Kiểm tra sự tồn tại của 2 file sống còn:
    -   `models/checkpoints/best_multimodal_cascade_binary_cross_attention.pt` (Chính - V3)
    -   `models/checkpoints/best_cascade_gnn_gat.pt` (Dự phòng)
3.  **Nguồn dữ liệu**: Chuẩn bị sẵn 3-5 Link Reddit thật và 1 file `.jsonl` mẫu.

---

## 🧪 GIAI ĐOẠN 1: KIỂM TRA ĐỘ SẴN SÀNG (HEALTH CHECK)

**Test Case 1.1: Trạng thái hệ thống qua API**
-   **Cách làm**: Chạy `python app.py` và truy cập `http://127.0.0.1:5000/api/predict/health`.
-   **Mục tiêu**: 
    -   `ready`: `true`
    -   `model_type`: `multimodal` (phải nhận diện đúng Ver 3).
    -   `classes`: `2` (hoặc `6` nếu dùng model cũ, hệ thống phải tự map kết quả).

---

## 🖼️ GIAI ĐOẠN 2: KIỂM THỬ ĐA PHƯƠNG THỨC (MULTIMODAL TESTING)

Đây là phần "ăn điểm" nhất, chứng minh model hiểu được cả ảnh và text.

**Test Case 2.1: Bài viết có Ảnh + Nhiều Comment (Mẫu chuẩn)**
-   **Input**: Link Reddit có ảnh rõ ràng.
-   **Kỳ vọng**: 
    -   `graph_info`: Nodes > 10, Edges > 10.
    -   `feature_dim`: `1793` (768 text + 1025 CLIP).
    -   `prediction`: Phản ánh đúng nội dung.

**Test Case 2.2: Bài viết "Treo đầu dê bán thịt chó" (Nâng cao)**
-   **Input**: Link có ảnh không liên quan đến tiêu đề (Ví dụ: Tiêu đề về chính trị nhưng ảnh là con mèo).
-   **Kỳ vọng**: CLIP Score thấp -> Model dựa vào sự mâu thuẫn này để tăng tỷ lệ `FAKE`.

---

## 🛠️ GIAI ĐOẠN 3: KIỂM THỬ ĐỘ BỀN (ROBUSTNESS & EDGE CASES)

Chứng minh hệ thống không bao giờ "chết" dù dữ liệu xấu.

**Test Case 3.1: Bài viết KHÔNG có ảnh (Text-only)**
-   **Input**: Bài viết dạng thảo luận (Self-post) không có link ảnh.
-   **Kỳ vọng**: Hệ thống log Warning `no_image_found`, tự động bù 0 cho vector ảnh, vẫn dự đoán dựa trên text và đồ thị lan truyền.

**Test Case 3.2: Bài viết MỚI (0 Comment)**
-   **Input**: Link bài viết cực mới, vừa đăng 1 phút.
-   **Kỳ vọng**: Đồ thị chỉ có 1 Node (Root), 0 Edge. Model vẫn chạy được (chế độ Global Mean Pool trên 1 node).

**Test Case 3.3: Link lỗi / 404**
-   **Input**: Nhập bừa một chuỗi không phải URL hoặc link đã bị xóa.
-   **Kỳ vọng**: Trả về `400 Bad Request` kèm thông báo lỗi rõ ràng thay vì lỗi `500 Server Error`.

---

## 📊 GIAI ĐOẠN 4: KIỂM THỬ HIỆU NĂNG THEO LÔ (BATCH TEST)

**Test Case 4.1: File JSONL lớn**
-   **Cách làm**: Upload file `test_demo.jsonl` có 50-100 bản ghi.
-   **Kỳ vọng**: 
    -   Xử lý tuần tự không bị tràn RAM.
    -   Hiển thị biểu đồ Accuracy/F1 Score ngay trên Dashboard.
    -   Báo cáo được số lượng bản ghi bị lỗi (Skipped count).

---

## 🆘 GIAI ĐOẠN 5: XỬ LÝ SỰ CỐ TẠI CHỖ (TROUBLESHOOTING)

Dành cho buổi thuyết trình nếu gặp lỗi:

1.  **Lỗi 403 Forbidden (Reddit chặn)**: 
    -   *Giải thích*: "Reddit đang tạm chặn IP do tần suất truy cập cao, tuy nhiên module trích xuất đặc trưng và đồ thị vẫn hoạt động bình thường."
    -   *Xử lý*: Chuyển sang dùng tính năng **Upload JSONL** (dữ liệu offline) để demo tiếp.
2.  **Lỗi Model load failed**:
    -   *Kiểm tra*: File checkpoint có nằm đúng thư mục không? Tên file có bị đổi không?
3.  **Lỗi GPU Out of Memory**:
    -   *Xử lý*: Khởi động lại `app.py`. Nếu vẫn bị, chỉnh `device: cpu` trong `predictor.py`.

---

## 🏁 TỔNG KẾT BẢNG ĐIỂM (DO-NOT-MISS)
- [ ] Phải show được Dashboard xanh mướt khi model Sẵn sàng.
- [ ] Phải giải thích được tại sao Feature Dim là **1793**.
- [ ] Phải chứng minh được Graph GNN giúp cải thiện độ chính xác so với chỉ dùng Text.
- [ ] Phải show được tính năng "Dự phòng" (Fallback) nếu model multimodal gặp sự cố.

---
*Người soạn: Antigravity AI Assistant*
