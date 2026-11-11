import sys
sys.path.append('.')
from src.data.reddit_crawler import RedditCrawler
import json
import time

def create_demo_batch():
    print("🚀 Đang khởi động Crawler để lấy dữ liệu Reddit mới nhất...")
    crawler = RedditCrawler(limit=5, subreddits=['science', 'TheOnion'])
    crawler._setup_session()
    
    batch_data = []
    
    for sub in crawler.subreddits:
        print(f"📡 Đang quét subreddit: r/{sub}...")
        try:
            url = f'https://www.reddit.com/r/{sub}/hot.json'
            resp = crawler.session.get(url, timeout=10)
            data = resp.json()
            
            for child in data['data']['children']:
                p = child['data']
                # Lọc bài có comment (>2) và bài không bị ghim
                if p['num_comments'] > 2 and not p.get('is_video', False) and not p.get('stickied', False):
                    permalink = p['permalink']
                    print(f"  - Lấy comment cho bài: {p['title'][:40]}...")
                    
                    # Gọi Crawler chính hiệu của bạn để lấy cây comment
                    cascade_nodes = crawler.fetch_comments(permalink)
                    
                    # Convert sang chuẩn item của model (predict_batch cần)
                    record = {
                        "id": p['id'],
                        "title": p.get('title', ''),
                        "raw_text": crawler.clean_text(p.get('title', ''), p.get('selftext', '')),
                        "timestamp": p.get('created_utc', int(time.time())),
                        "media_url": p.get('url_overridden_by_dest') or p.get('url', ''),
                        "cascade": cascade_nodes,
                        "metadata": {"subreddit": sub, "permalink": permalink}
                    }
                    batch_data.append(record)
                    
                    # Giới hạn 5 bài mỗi Sub để demo chạy cho lẹ
                    if len([x for x in batch_data if x['metadata']['subreddit'] == sub]) >= 5:
                        break
        except Exception as e:
            print(f"❌ Lỗi khi tải r/{sub}: {e}")

    # Xả dữ liệu ra file JSONL
    out_file = "test_demo_batch.jsonl"
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in batch_data:
            f.write(json.dumps(item) + '\n')
            
    print(f"\n✅ Đã tạo xong file {out_file} chứa {len(batch_data)} bài đăng MỚI TINH!")
    print("👉 Chiều nay hãy dùng tab 'Batch Predict' và upload file này lên để Demo.")

if __name__ == "__main__":
    create_demo_batch()
