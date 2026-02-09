import json
import os
import time
from reddit_crawler import RedditCrawler 

# Cấu hình đường dẫn
INPUT_FILE = "data/labeled_master.jsonl"
OUTPUT_FILE = "data/reddit_enriched_data.jsonl"

def reddit_enriched_data():
    # Khởi tạo crawler từ file gốc 
    crawler = RedditCrawler(debug=False)

    if not os.path.exists(INPUT_FILE):
        print(f"Khong tim thay file {INPUT_FILE}")
        return

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    processed_count = 0

    # Đọc dữ liệu đã gắn label
    print(f"Dang doc du lieu tu {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        target_posts = [json.loads(line) for line in f]

    print(f"Tim thay {len(target_posts)} bài viết cần làm giàu dữ liệu.")
    
    for index, post in enumerate(target_posts):
        post_id = post.get('id')
        fake_permalink = f"/comments/{post_id}/"

        print(f"[{index + 1}/{len(target_posts)}] Đang lấy Cascade cho: {post_id}...")
        
        try:
            # Gọi hàm fetch comments có sẵn
            cascade_data = crawler.fetch_comments(fake_permalink)
            
            # Gộp dữ liệu cascade vào object gốc
            post['cascade'] = cascade_data
            post['metadata_enrich'] = {
                "enriched_at": int(time.time()),
                "comment_count_fetched": len(cascade_data)
            }

            # Lưu ngay dữ liệu (chế độ 'a' - append)
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
                out_file.write(json.dumps(post, ensure_ascii=False) + '\n')
            
            processed_count += 1

        except Exception as e:
            print(f"Lỗi khi xử lý {post_id}: {e}")

        # Thời gian nghỉ sau mỗi bài để tránh bị Reddit chặn (Rate Limit)
        time.sleep(2.0) 

    print(f"Hoàn thành! Đã xử lý thành công {processed_count}/{len(target_posts)} bài viết.")
    print(f"Dữ liệu được lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    reddit_enriched_data()