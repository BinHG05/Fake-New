import json
import os
import time
import sys

# Thêm đường dẫn để Python tìm thấy module dù chạy từ root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

try:
    from reddit_crawler import RedditCrawler 
except ImportError:
    # Nếu không tìm thấy, thử import theo absolute path (phòng hờ)
    try:
        from src.data.reddit_crawler import RedditCrawler
    except ImportError:
        print("❌ Lỗi: Không tìm thấy module reddit_crawler.py")
        sys.exit(1)

# Sử dụng đường dẫn tuyệt đối dựa trên vị trí file hiện tại
# Lên 2 cấp: src/data -> src -> ProjectRoot
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
INPUT_FILE = os.path.join(ROOT_DIR, "data", "03_clean", "Fakeddit", "labeled_master.jsonl")
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "reddit_enriched_data.jsonl")

# Hàm kiểm tra các ID đã xử lý
def get_existing_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    existing_ids = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(data['id'])
                except:
                    continue
    except Exception as e:
        print(f"⚠️ Warning: Lỗi khi đọc file cũ: {e}")
    return existing_ids

def reddit_enriched_data():
    print(f"🚀 Bắt đầu quá trình Enrich Data...")
    
    # Khởi tạo crawler từ file gốc 
    crawler = RedditCrawler(debug=False)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy file input: {INPUT_FILE}")
        print(f"👉 Hãy đảm bảo file 'labeled_master.jsonl' nằm trong thư mục 'data' của project.")
        return

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Lấy danh sách ID đã hoàn thành để Resume
    done_ids = get_existing_ids(OUTPUT_FILE)
    print(f"🔄 Đã xử lý xong {len(done_ids)} bài viết trước đó.")

    # Đọc dữ liệu đã gắn label
    print(f"📖 Đang đọc dữ liệu từ {INPUT_FILE}...")
    target_posts = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    target_posts.append(json.loads(line))
                except:
                    continue
    except Exception as e:
        print(f"❌ Lỗi đọc file input: {e}")
        return

    total_posts = len(target_posts)
    print(f"✅ Tìm thấy {total_posts} bài viết cần xử lý.")
    
    processed_count = 0
    skipped_count = 0
    
    for index, post in enumerate(target_posts):
        post_id = post.get('id')
        
        # Safety check: Nếu không có ID thì bỏ qua
        if not post_id:
            continue

        # Kiểm tra Resume (Nếu ID đã có thì bỏ qua)
        if post_id in done_ids:
            skipped_count += 1
            # Chỉ in log mỗi 50 bài skip để đỡ spam màn hình
            if skipped_count % 50 == 0:
                print(f"⏩ Đã bỏ qua {skipped_count} bài cũ...", end='\r')
            continue

        fake_permalink = f"/comments/{post_id}/"
        print(f"📥 [{index + 1}/{total_posts}] Đang lấy Cascade cho: {post_id}...")
        
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
            print(f"   ✅ OK! Lấy được {len(cascade_data)} comments.")

        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {post_id}: {e}")

        # Thời gian nghỉ để tránh Rate Limit
        time.sleep(2.0) 

    print("\n" + "="*50)
    print(f"🏁 XONG!")
    print(f"📊 Tổng cộng: {total_posts}")
    print(f"✅ Mới làm xong: {processed_count}")
    print(f"⏩ Đã bỏ qua: {skipped_count}")
    print(f"💾 File kết quả: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        # Fix encoding cho Windows terminal
        if sys.platform.startswith('win'):
            os.system('chcp 65001')
    except:
        pass
    reddit_enriched_data()