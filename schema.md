SCHEMA DỮ LIỆU DỰ ÁN: AI PHÁT HIỆN TIN GIẢ

Mục tiêu: Tài liệu này định nghĩa cấu trúc dữ liệu tiêu chuẩn (Schema) để đồng nhất việc lưu trữ và xử lý các thông tin Text, Image, Graph, và Cascade xuyên suốt dự án.

Công cụ & Phương pháp

- Dữ liệu được lưu trữ ở định dạng JSONL (JSON Lines).
- Timestamp phải được chuẩn hóa về định dạng Epoch (Integer).

1. Sơ Đồ Cấu Trúc Dữ Liệu Tổng Thể
   Dữ liệu được tổ chức thành 4 bảng chính để hỗ trợ mô hình Multimodal (Text/Image) và Graph Neural Network (GNN).
   ![alt text](image.png)

- POST: Chứa nội dung chính, nhãn (label), và các features của bài đăng.
- USER: Chứa thông tin hồ sơ người dùng (user_id).
- MEDIA: Chứa thông tin file ảnh/video đã xử lý.
- EDGES (Cascade/Graph): Mô tả quan hệ lan truyền (retweet, reply) giữa các post và user.

2.  CORE SCHEMA (Đầu Vào Thô)
    Đây là các trường bắt buộc mà Thành viên A phải cung cấp trong file metadata_raw.jsonl từ các nguồn Kaggle/Github (LIAR, Fakeddit, FakeNewsNet).

    id:

    - Type: String
    - Description: ID duy nhất của bài đăng (ví dụ: TW_1001).
    - Ghi chú: Khóa chính

    timestamp:

    - Type: Integer (Epoch)
    - Description: Thời gian đăng bài gốc.
    - Ghi chú: Phải chuẩn hóa về Epoch.

    label:

    - Type: String
    - Description: Nhãn gốc (ví dụ: "Fake", "True", "Mostly True").

    raw_text:

    - Type: String
    - Description: Văn bản gốc, chưa làm sạch (có chứa URL, emoji...).

    media_url:

    - Type: String
    - Description: URL tới tệp ảnh/video gốc.

    user_id:

    - Type: String
    - Description: ID của người dùng đăng bài.

    retweet_count:

    - Type: Integer
    - Description: Số lần retweet/chia sẻ gốc (ví dụ: 500).

3.  EXTENDED SCHEMA (Đầu Ra Đã Xử Lý)
    Đây là các trường features được tính toán bởi Thành viên B (Text) và C (Media). Các trường raw_text và media_url thô sẽ được thay thế bằng các trường đã xử lý.
    3.1. Text Processing Fields (Thành viên B)

    clean_text

    - Type: String

    - Description: Văn bản đã làm sạch (xóa URL, hashtag, @mention), chuyển thành chữ thường (lowercase).

    - Responsibility: B

    text_features

    - Type: Object

    - Description: Tập hợp các features thống kê văn bản.

    - Responsibility: B

    word_count

    - Type: Integer

    - Description: Tổng số từ trong clean_text.

    - Responsibility: B

    has_caps_lock

    - Type: Boolean

    - Description: True nếu tỷ lệ chữ hoa lớn hơn 50% trong raw_text.

    - Responsibility: B

    sentiment_score

    - Type: Float

    - Description: Điểm cảm xúc (-1.0 đến 1.0).

    - Responsibility: B

      3.2. Media Processing Fields (Thành viên C)

      image_info

      - Type: Object

      - Description: Thông tin về tệp media đã xử lý.

      - Responsibility: C

      processed_path

      - Type: String

      - Description: Đường dẫn tương đối tới file ảnh đã resize (ví dụ: data/processed/images/TW_1001.jpg).

      - Responsibility: C

      image_size

      - Type: Array [Int, Int]

      - Description: Kích thước ảnh sau khi resize. Phải là [224, 224] cho mô hình CV.

      - Responsibility: C

      is_video

      - Type: Boolean

      - Description: True nếu bài đăng có video.

      - Responsibility: C

      keyframe_paths

      - Type: Array [String]

      - Description: Danh sách đường dẫn tới các keyframe (8-16 khung hình) đã trích xuất từ video.

      - Responsibility: C

        3.3 Graph & User Features (Thành viên D / Giai đoạn sau)
        user_features

        - Type: Object

        - Description: Đặc trưng của người dùng đăng bài.

        - Responsibility: D/A/Graph Team

        account_age_days

        - Type: Integer

        - Description: Tuổi của tài khoản tính bằng ngày.

        - Responsibility: D/A

        user_reputation_score

        - Type: Float

        - Description: Điểm uy tín người dùng (D cần định nghĩa công thức dựa trên followers, activity).

        - Responsibility: D

4.  QUY TẮC KIỂM TRA (VALIDATOR RULES)
    Thành viên D phải lập trình validate_schema.py để thực thi các quy tắc sau trên mọi lô dữ liệu.

- Tính Đầy Đủ: Các trường id, label, clean_text, và processed_path phải tồn tại trong mọi bản ghi.

- Kiểu Dữ Liệu:

* timestamp, word_count phải là Integer.

* sentiment_score, user_reputation_score phải là Float.

- Quy tắc Cấu trúc: image_size phải là một mảng có đúng 2 phần tử số nguyên, và giá trị phải là [224, 224].

- Tính Nhất Quán (Consistency):

* Các user_id trong bảng POST phải tồn tại trong bảng USER.

* Kiểm tra sự trùng lặp của id (mỗi bài đăng chỉ có một ID duy nhất).

5. Minh Họa JSON (Ví dụ)
   A. Core Schema (Đầu vào thô)
   {
   "id": "TW_1001",
   "timestamp": 1672531200,
   "label": "Fake",
   "raw_text": "BREAKING: Aliens landed in Time Square! Check link: http://bit.ly/123 #ufo",
   "media_url": "http://twitter.com/img/ufo_fake.jpg",
   "user_id": "U_456"
   }
   B. Extended Schema (Đầu ra đã xử lý)
   {
   "id": "TW_1001",
   "timestamp": 1672531200,
   "label": "Fake",
   "clean_text": "breaking aliens landed in time square check link",
   "text_features": {
   "word_count": 8,
   "has_caps_lock": true,
   "sentiment_score": -0.5
   },
   "image_info": {
   "processed_path": "data/processed/images/TW_1001.jpg",
   "image_size": [224, 224],
   "is_video": false,
   "keyframe_paths": []
   },
   "graph_features": {
   "is_viral": true
   }
   }
