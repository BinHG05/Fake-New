import json
import os
import time
import sys
# Thêm đường dẫn để Python tìm thấy module dù chạy từ root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reddit_crawler import RedditCrawler 

# Sử dụng đường dẫn tuyệt đối dựa trên vị trí file hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Lên 2 cấp: src/data -> src -> root
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
INPUT_FILE = os.path.join(ROOT_DIR, "data", "labeled_master.jsonl")
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "reddit_enriched_data.jsonl")

# Hàm kiểm tra các ID đã xử lý
def get_existing_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    existing_ids = set()
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                existing_ids.add(data['id'])
            except:
                continue
    return existing_ids

def reddit_enriched_data():
    # Khởi tạo crawler từ file gốc 
    crawler = RedditCrawler(debug=False)

    if not os.path.exists(INPUT_FILE):
        print(f"Khong tim thay file {INPUT_FILE}")
        return

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Lấy danh sách ID đã hoàn thành để Resume
    done_ids = get_existing_ids(OUTPUT_FILE)
    print(f"Da xu ly xong {len(done_ids)} bai viet truoc do.")

    # Đọc dữ liệu đã gắn label
    print(f"Dang doc du lieu tu {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        target_posts = [json.loads(line) for line in f]

    print(f"Tim thay {len(target_posts)} bai viet can lam giàu du lieu.")
    
    processed_count = 0
    for index, post in enumerate(target_posts):
        post_id = post.get('id')
        
        # Kiểm tra Resume (Nếu ID đã có thì bỏ qua)
        if post_id in done_ids:
            # In rõ [SKIP] để theo dõi tiến độ
            print(f"[{index + 1}/{len(target_posts)}] [SKIP] ID: {post_id} da ton tai.")
            continue

        fake_permalink = f"/comments/{post_id}/"
        print(f"[{index + 1}/{len(target_posts)}] Dang lay Cascade cho: {post_id}...")
        
        try:
            # Gọi hàm fetch comments có sẵn
            cascade_data = crawler.fetch_comments(fake_permalink)
            
            # Gộp dữ liệu cascade vào object gốc
            post['cascade'] = cascade_data
            post['metadata_enrich'] = {
                "enriched_at": int(time.time()),
                "comment_count_fetched": len(cascade_data)
            }

            # Lưu ngay dữ liệu (mode 'a' - append)
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_file:
                out_file.write(json.dumps(post, ensure_ascii=False) + '\n')
            
            processed_count += 1

        except Exception as e:
            print(f"Loi khi xu li {post_id}: {e}")

        # Thời gian nghỉ để tránh Rate Limit
        time.sleep(2.0) 

    print(f"Hoanh thanh! Da xu ly them {processed_count} bai viet moi.")
    print(f"Du lieu duoc luu tai: {OUTPUT_FILE}")

if __name__ == "__main__":
    reddit_enriched_data()