import requests
import json
import time
from datetime import datetime
import os

# --- CẤU HÌNH ---
# 1. Danh sách các nhóm cần quét tin mới (Real-time)
SUBREDDITS = ["worldnews", "news", "politics", "technology", "conspiracy", "fake_news"]

# 2. Tự động xác định đường dẫn để lưu file vào thư mục 'data' ở ngoài cùng
# Logic: Từ file này (src/data) đi ngược ra 2 cấp là tới thư mục gốc -> vào folder data
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục chứa file code này
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR)) # Thư mục gốc dự án (FAKE-NEW)
DATA_DIR = os.path.join(ROOT_DIR, "data") # Thư mục kho chứa data

# Tạo tên file kết quả
OUTPUT_FILE = os.path.join(DATA_DIR, "reddit_realtime_data.jsonl")

def crawl_reddit_realtime():
    print(f"🚀 BẮT ĐẦU QUÉT DỮ LIỆU REAL-TIME")
    print(f"📂 File sẽ được lưu tại: {OUTPUT_FILE}")
    print("-" * 50)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_posts = []

    for group in SUBREDDITS:
        try:
            print(f"📡 Đang quét nhóm: r/{group}...")
            # Lấy dữ liệu JSON công khai (Cửa sau - Không cần API Key)
            url = f"https://www.reddit.com/r/{group}/new.json?limit=20"
            
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                posts = data['data']['children']
                
                count = 0
                for post in posts:
                    p = post['data']
                    
                    # Map dữ liệu sang đúng Schema JSONL của nhóm
                    item = {
                        "id": p['id'],
                        "timestamp": p['created_utc'],  # Thời gian thực
                        "label": "Unlabeled",           # Chưa có nhãn
                        "raw_text": f"{p['title']} {p['selftext']}", # Tiêu đề + Nội dung
                        "media_url": p.get('url_overridden_by_dest', None), # Link ảnh (nếu có)
                        "user_id": p['author'],         # Tác giả
                        "source": f"r/{group}"
                    }
                    all_posts.append(item)
                    count += 1
                print(f"   ✅ Lấy được {count} bài mới.")
            else:
                print(f"   ⚠️ Lỗi kết nối r/{group}: {resp.status_code}")
                
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            
        time.sleep(1) # Nghỉ 1 giây để không bị chặn

    # Lưu file
    if all_posts:
        print("-" * 50)
        print(f"💾 Đang lưu {len(all_posts)} dòng dữ liệu...")
        
        # Đảm bảo thư mục data tồn tại
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for post in all_posts:
                json.dump(post, f, ensure_ascii=False)
                f.write('\n')
                
        print("🎉 HOÀN TÀNH! NHIỆM VỤ XONG.")
    else:
        print("😭 Không lấy được dữ liệu nào.")

if __name__ == "__main__":
    crawl_reddit_realtime()