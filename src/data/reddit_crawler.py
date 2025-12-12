import requests
import json
import time
import os

# --- CẤU HÌNH ---
SUBREDDITS = ["worldnews", "news", "politics", "technology", "conspiracy", "fake_news"]

# Tự động xác định đường dẫn lưu vào folder 'data' ở ngoài cùng
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_realtime_data.jsonl")

# --- HÀM BỊ THIẾU TRONG ĐOẠN CỦA BẠN (QUAN TRỌNG) ---
def get_existing_ids(file_path):
    """Đọc file cũ để lấy danh sách ID đã có (tránh lưu trùng)"""
    existing_ids = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'id' in record:
                            existing_ids.add(record['id'])
                    except:
                        continue
        except:
            pass
    return existing_ids
# ----------------------------------------------------

def crawl_reddit_final():
    print(f"🚀 BẮT ĐẦU QUÉT DỮ LIỆU REAL-TIME (SCHEMA CHUẨN)")
    print(f"📂 File lưu tại: {OUTPUT_FILE}")
    print("-" * 50)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # 1. Load ID cũ để lọc trùng
    existing_ids = get_existing_ids(OUTPUT_FILE)
    print(f"📊 Trong kho đang có: {len(existing_ids)} bài.")
    
    buffer_to_write = []

    for group in SUBREDDITS:
        try:
            print(f"📡 Đang quét nhóm: r/{group}...")
            url = f"https://www.reddit.com/r/{group}/new.json?limit=25"
            
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                children = data['data']['children']
                
                count = 0
                for child in children:
                    p = child['data']
                    post_id = str(p['id'])

                    # Check trùng: Nếu ID đã có trong kho thì bỏ qua
                    if post_id in existing_ids:
                        continue

                    # Xử lý media_url: Phải là String rỗng "" nếu không có ảnh
                    media = p.get('url_overridden_by_dest', "")
                    if media is None: 
                        media = ""

                    # --- MAP ĐÚNG CHUẨN SCHEMA ---
                    item = {
                        "id": post_id,
                        "timestamp": int(p['created_utc']),
                        "label": "Unlabeled",
                        "raw_text": f"{p['title']} {p['selftext']}".strip(),
                        "media_url": str(media),
                        "user_id": str(p['author']),
                        "retweet_count": int(p.get('num_comments', 0))
                    }

                    buffer_to_write.append(item)
                    existing_ids.add(post_id) 
                    count += 1
                
                print(f"   ✅ Lấy được {count} bài mới.")
            else:
                print(f"   ⚠️ Lỗi kết nối r/{group}: {resp.status_code}")
                
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            
        time.sleep(2) # Nghỉ 2s tránh chặn IP

    # 2. Lưu xuống file (Mode 'a' - Append để cộng dồn)
    if buffer_to_write:
        print("-" * 50)
        
        # Đảm bảo thư mục data tồn tại
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for post in buffer_to_write:
                json.dump(post, f, ensure_ascii=False)
                f.write('\n')
                
        print(f"🎉 THÀNH CÔNG! Đã lưu thêm {len(buffer_to_write)} bài viết mới.")
    else:
        print("😴 Không có bài mới nào so với lần chạy trước.")

if __name__ == "__main__":
    crawl_reddit_final()