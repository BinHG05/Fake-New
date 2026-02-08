# Phase 2B: Cascade Data Collection Implementation Plan

## Mục tiêu
Nâng cấp `reddit_crawler.py` để thu thập **toàn bộ cây thảo luận (comment tree)** của mỗi bài viết. Dữ liệu này là nguyên liệu bắt buộc để xây dựng **Cascade Graph** (Mô hình lan truyền tin tức).

## Lý do cần thiết
Hiện tại crawler chỉ lấy thông tin bài gốc (Post). Để phân tích tin giả, ta cần biết:
- Ai là người phản hồi? (User node)
- Phản hồi lúc nào? (Temporal edge)
- Phản hồi cho ai? (Structure)

## Thiết kế Kỹ thuật

### 1. Data Schema Mới
Mỗi dòng JSONL sẽ có thêm trường `cascade`:

```json
{
  "id": "post_id",
  "title": "...",
  "cascade": [
    {
      "id": "comment_id_1",
      "parent_id": "post_id",
      "user_id": "user_A",
      "timestamp": 1234567890,
      "text": "Fake news!",
      "level": 1
    },
    {
      "id": "comment_id_2",
      "parent_id": "comment_id_1",
      "user_id": "user_B",
      "timestamp": 1234567999,
      "text": "No it is not",
      "level": 2
    }
  ]
}
```

### 2. Thuật toán Crowling (DFS Recursive)
Sử dụng Reddit JSON API: `https://www.reddit.com{permalink}.json`

- **Input:** Permalink của bài post.
- **Process:**
    - Request JSON.
    - Parse object `data` -> `children` -> `replies`.
    - Dùng đệ quy để duyệt hết các node con (replies of replies).
- **Constraints:**
    - Rate limit: Cần `time.sleep` hợp lý.
    - Depth limit: Giới hạn độ sâu để tránh loop vô hạn.

## Proposed Changes

### [MODIFY] [reddit_crawler.py](file:///d:/NCKH_Project/Project/src/data/reddit_crawler.py)
- Thêm method `fetch_comments(self, permalink)`
- Thêm method `parse_comment_tree(self, comment_data, parent_id)`
- Cập nhật hàm `crawl()` để gọi `fetch_comments` cho mỗi bài viết.

## Verification Plan
1. Chạy crawler với limit nhỏ (5 bài).
2. Kiểm tra file output xem có trường `cascade` không.
3. Kiểm tra cấu trúc phân cấp (parent_id) có đúng logic không.
