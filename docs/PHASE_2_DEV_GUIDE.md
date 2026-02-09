# 🧑‍💻 HƯỚNG DẪN KỸ THUẬT PHASE 2: CASCADE DATA COLLECTION

Tài liệu này dành cho thành viên phụ trách **Giai đoạn 2**.
Nhiệm vụ của bạn không chỉ là chạy code, mà phải hiểu **tại sao lại có các file này** và **chúng hoạt động thế nào**.

---

## 🗺️ Bản đồ các file (File Map)

Trong folder `src/features/`, bạn sẽ thấy nhiều file, nhưng ở Phase 2 này, bạn chỉ cần quan tâm đúng **3 file** này thôi:

1.  **`src/data/reddit_crawler.py`** (Crawler - Người đi lấy tin)
2.  **`src/features/embedding_extractor.py`** (Phụ trợ - Bộ mã hóa)
3.  **`src/features/cascade_graph_builder.py`** (Builder - Kiến trúc sư)

*(Các file khác như `graph_builder.py` là của Phase sau, bạn cứ lờ đi).*

---

## 🔍 GIẢI PHẪU CHI TIẾT TỪNG FILE

### 1. `reddit_crawler.py` (Crawler)
*   **Nhiệm vụ:** Đi lên Reddit, tìm bài viết -> tải luôn cả phần bình luận (Reply Tree).
*   **Tại sao cần?**
    *   Model NCKH của mình cần biết "Tin này lan truyền thế nào?".
    *   Ví dụ: A đăng tin -> B vào chửi -> C vào bênh vực. Cấu trúc A->B->C chính là "Cascade".
### 🛠️ NHIỆM VỤ CỦA BẠN (ACTION ITEMS)

Hiện tại file `reddit_crawler.py` mới chỉ demo việc lấy bài mới. Bạn cần cải tiến nó để phục vụ 2 mục đích:

1.  **Mục đích 1: Lấy dữ liệu mới (New Data)**
    *   Crawl các bài viết mới nhất từ Reddit về.
    *   Lưu ý: Dữ liệu này chưa có nhãn -> Cần đẩy lên Label Studio.

2.  **Mục đích 2: Làm giàu dữ liệu cũ (Enrich Data)**
    *   Đọc file `data/labeled_master.jsonl` (đã có nhãn True/Fake từ trước).
    *   Lấy ID bài viết -> Gọi API Reddit để lấy thêm Comment Tree.
    *   Gộp Comment vào dữ liệu cũ -> Xuất ra file mới.

💡 **Gợi ý cách làm:**
Bạn có quyền quyết định cách code, miễn sao chạy được:
*   **Cách A (Gộp):** Sửa `reddit_crawler.py`, thêm tham số `--enrich [file_path]` để chuyển chế độ.
*   **Cách B (Tách):** Tạo thêm file `crawler_enrich.py` chuyên dụng, copy logic từ crawler gốc sang và sửa đổi.

Hãy chọn cách bạn thấy tối ưu nhất!

### 2. `embedding_extractor.py` (Bộ mã hóa - Quan trọng)
*   **Nhiệm vụ:** Biến chữ (Text) và ảnh (Image) thành các con số (Vector hay Embedding).
*   **Tại sao cần?**
    *   Máy tính không hiểu chữ "Fake News", nó chỉ hiểu số `[0.12, -0.56, ...]`.
    *   File này dùng **XLM-RoBERTa** (cho text) và **CLIP** (cho ảnh) để làm việc đó.
*   **Code hoạt động thế nào?**
    *   Nó load model nặng (vài trăm MB) vào RAM/GPU.
    *   Có hàm `extract(text)`: Nhận vào câu nói -> Trả về vector 768 chiều.
*   **Việc bạn cần làm:**
    *   **Không cần sửa code file này.**
    *   Nhưng phải hiểu: "À, cái file `cascade_graph_builder.py` tí nữa sẽ gọi file này để nhờ dịch tiếng người sang tiếng máy".

### 3. `cascade_graph_builder.py` (Builder - Trùm cuối Phase 2)
*   **Nhiệm vụ:** Kết hợp dữ liệu từ (1) và công cụ từ (2) để xây nên **Đồ thị (Graph)**.
*   **Tại sao cần?**
    *   Đây là bước chuẩn bị nguyên liệu cho Graph Neural Network (GNN).
    *   Nếu không có file này, GNN không có gì để ăn (train).
*   **Logic:**
    1.  **Node:** Mỗi comment là 1 chấm tròn.
    2.  **Featues:** Dùng `embedding_extractor` để mã hóa nội dung comment thành vector số.
    3.  **Edge:** Dùng `networkx` để nối dây. Nếu B reply A -> Tạo cạnh nối A -> B.
*   **Việc bạn cần làm:**
    *   Viết script (hoặc dùng notebook) import class này vào.
    *   Gọi hàm `process_dataset(items)` để nó chạy vòng lặp chuyển đổi toàn bộ dữ liệu ra file `.pt`.

---

## ✅ Checklist cho bạn (Definition of Done)

Để hoàn thành nhiệm vụ Phase 2, bạn cần trả về kết quả:

1.  [ ] Một folder chứa các file `.pt` (mỗi file là một đồ thị bài viết).
2.  [ ] Chắc chắn rằng trong file `.pt` đó có:
    *   `x`: Chứa đặc trưng của bài viết và comment (đã mã hóa thành số).
    *   `edge_index`: Chứa cấu trúc ai trả lời ai.

Nếu bạn hiểu 3 file trên và chạy ra được file `.pt`, coi như bạn đã làm chủ giai đoạn này! 🚀
