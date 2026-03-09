import requests
import json
import time
import os
import re
import logging
import argparse
import uuid
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
CRAWLER_VERSION = "1.1.0"
SUBREDDITS = ["worldnews", "news", "politics", "technology", "conspiracy"]
# Subreddits có nhiều hình ảnh hơn cho multimodal training
SUBREDDITS_IMAGE_HEAVY = [
    "pics", "memes", "PoliticalHumor", "facepalm", "therewasanattempt",
    "Damnthatsinteresting", "interestingasfuck", "MurderedByWords",
    "confidentlyincorrect", "quityourbullshit",
    "propaganda", "mediamanipulation", "fakefacts"
]
USER_REGEX = r"^[a-zA-Z0-9_-]{3,20}$"
MEDIA_EXTENSIONS = r"\.(jpg|jpeg|png|gif|mp4|webm|mov)$"
VIDEO_EXTENSIONS = r"\.(mp4|webm|mov|gifv)(\?|$)"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
OUTPUT_FILE = os.path.join(DATA_DIR, "01_raw", "reddit", "reddit_realtime_data.jsonl")

# --- LOGGING SETUP ---
def setup_logging(debug=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_level = logging.DEBUG if debug else logging.INFO
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Crawl log (Rotation: 5MB per file, keep 3)
    crawl_handler = RotatingFileHandler(os.path.join(LOG_DIR, "crawl.log"), maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    crawl_handler.setLevel(log_level)
    crawl_handler.setFormatter(formatter)
    
    # Error log
    error_handler = RotatingFileHandler(os.path.join(LOG_DIR, "error.log"), maxBytes=2*1024*1024, backupCount=5, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger("RedditCrawler")
    logger.setLevel(log_level)
    logger.addHandler(crawl_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger

class RedditCrawler:
    def __init__(self, debug=False, limit=25, subreddits=None, images_only=False, min_text_length=10):
        self.logger = setup_logging(debug)
        self.debug = debug
        self.limit = limit
        self.subreddits = subreddits or SUBREDDITS
        self.images_only = images_only
        self.min_text_length = min_text_length
        self.run_id = str(uuid.uuid4())[:8]
        self.session = self._setup_session()
        self.stats = {"processed": 0, "new": 0, "skipped_empty": 0, "skipped_text_too_short": 0, "skipped_video": 0, "skipped_no_image": 0, "errors": 0}
        
    def _setup_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityCrawler/1.1'})
        return session

    def get_existing_ids(self):
        existing_ids = set()
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if 'id' in record:
                                existing_ids.add(record['id'])
                        except: continue
            except Exception as e:
                self.logger.error(f"Error reading existing IDs: {e}")
        return existing_ids

    def clean_text(self, title, selftext):
        # 1. Strip Unicode control characters
        text = f"{title} {selftext}"
        text = "".join(ch for ch in text if ord(ch) >= 32 or ch == '\n')
        
        # 2. Handle [deleted] / [removed]
        if "[deleted]" in text or "[removed]" in text:
            text = text.replace("[deleted]", "").replace("[removed]", "").strip()
            
        # 3. Strip whitespace and limit length
        text = text.strip()
        if len(text) > 2000:
            text = text[:1997] + "..."
            
        return text

    def _preview_clean_text(self, text):
        """
        Mô phỏng logic clean_text của downstream pipeline (fakeddit_process_text.py)
        để dự đoán sớm text có bị loại không.
        """
        if not text:
            return ""
        # Xóa URL (giống downstream)
        preview = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Xóa special chars (giống downstream)
        preview = re.sub(r'[^\w\s.,!?\'\-]', '', preview)
        # Normalize whitespace
        preview = ' '.join(preview.split()).strip()
        return preview

    def standardize_user(self, author):
        if not author or author in ["[deleted]", "[removed]"]:
            return "unknown_user"
        if not re.match(USER_REGEX, author):
            return "unknown_user"
        return author

    def classify_media(self, url, post_data=None):
        if not url:
            # Fallback: chỉ lấy direct URL từ i.redd.it (không dùng preview/thumbnail)
            if post_data:
                preview = post_data.get('preview', {})
                images = preview.get('images', [])
                if images:
                    source_url = images[0].get('source', {}).get('url', '')
                    if source_url:
                        source_url = source_url.replace('&amp;', '&')
                        # Chỉ dùng nếu là i.redd.it (direct link, không bị 403)
                        if 'i.redd.it' in source_url:
                            return source_url
                # KHÔNG dùng preview.redd.it hay thumbnail (luôn bị 403/expired)
            return ""
        
        # Direct media (i.redd.it, imgur) - có extension rõ ràng
        if re.search(MEDIA_EXTENSIONS, url, re.IGNORECASE):
            # Chỉ chấp nhận domain đáng tin cậy
            if 'i.redd.it' in url or 'imgur.com' in url or 'i.imgur.com' in url:
                return url
        
        # Reddit hosted images (i.redd.it without extension)
        if 'i.redd.it' in url:
            return url
        
        # SKIP: preview.redd.it và external-preview.redd.it (luôn bị 403 Forbidden)
        # Token trong URL chỉ valid tạm thời, download sau sẽ fail
            
        # Imgur links (often without extension)
        if 'imgur.com' in url and '/a/' not in url and '/gallery/' not in url:
            # Add .jpg if no extension
            if not re.search(r'\.[a-zA-Z]{3,4}$', url):
                return url + '.jpg'
            return url
        
        return ""

    def fetch_comments(self, permalink):
        """
        Fetch the full comment tree for a given post permalink.
        Returns a list of comment objects (flat structure with parent_id).
        """
        if not permalink:
            return []

        # Ensure permalink has trailing slash
        if not permalink.endswith("/"):
            permalink += "/"
            
        url = f"https://www.reddit.com{permalink}.json?limit=100" # Limit top-level comments
        
        try:
            time.sleep(1.0) # Rate limit for comment fetch
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 429:
                self.logger.warning("   ⚠️ Rate limited (429) on comments. Waiting 5s...")
                time.sleep(5)
                return []
            resp.raise_for_status()
            
            data = resp.json()
            # data[0] is the post, data[1] is the comments Listing
            if len(data) < 2:
                return []
                
            comment_listing = data[1]
            comments_data = comment_listing.get('data', {}).get('children', [])
            
            cascade_nodes = []
            post_id = data[0]['data']['children'][0]['data']['id']

            for comment in comments_data:
                self.parse_comment_tree(comment, post_id, cascade_nodes, level=1)
                
            return cascade_nodes
            
        except Exception as e:
            self.logger.error(f"   ⚠️ Failed to fetch comments for {permalink}: {e}")
            return []

    def parse_comment_tree(self, comment_data, parent_id, cascade_nodes, level):
        """
        Recursive function to traverse comment replies.
        """
        kind = comment_data.get('kind', '')
        data = comment_data.get('data', {})
        
        # 't1' is a comment, 'more' is a "load more" button (skip for now)
        if kind != 't1': 
            return

        comment_id = data.get('id')
        if not comment_id:
            return

        body = self.clean_text("", data.get('body', ''))
        if not body:
            return

        node = {
            "id": comment_id,
            "parent_id": parent_id,
            "user_id": self.standardize_user(data.get('author', '')),
            "timestamp": int(data.get('created_utc', 0)),
            "text": body,
            "level": level
        }
        cascade_nodes.append(node)

        # Recursively process replies
        replies = data.get('replies', "")
        if isinstance(replies, dict): # If there are replies
            children = replies.get('data', {}).get('children', [])
            for child in children:
                self.parse_comment_tree(child, comment_id, cascade_nodes, level + 1)

    def crawl(self):
        self.logger.info(f"🚀 Starting Reddit Crawl (Run ID: {self.run_id}, Version: {CRAWLER_VERSION})")
        self.logger.info(f"📋 Subreddits: {', '.join(self.subreddits)}")
        if self.images_only:
            self.logger.info(f"🖼️  Mode: IMAGES ONLY (bỏ qua bài không có ảnh)")
        existing_ids = self.get_existing_ids()
        self.logger.info(f"📊 Current database size: {len(existing_ids)} items.")
        
        all_new_items = []
        
        for group in self.subreddits:
            try:
                self.logger.info(f"📡 Crawling r/{group}...")
                url = f"https://www.reddit.com/r/{group}/new.json?limit={self.limit}"
                
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                
                data = resp.json()
                children = data.get('data', {}).get('children', [])
                
                group_count = 0
                for child in children:
                    p = child.get('data', {})
                    post_id = str(p.get('id', ''))
                    
                    if not post_id or post_id in existing_ids:
                        continue
                        
                    raw_text = self.clean_text(p.get('title', ''), p.get('selftext', ''))
                    if not raw_text:
                        self.stats["skipped_empty"] += 1
                        continue
                    
                    # --- EARLY FILTER: Text quality (mô phỏng downstream) ---
                    preview_text = self._preview_clean_text(raw_text)
                    if len(preview_text) < self.min_text_length:
                        self.stats["skipped_text_too_short"] += 1
                        continue
                        
                    # Check timestamp sanity
                    crawl_time = int(time.time())
                    post_time = int(p.get('created_utc', 0))
                    if post_time > crawl_time:
                        post_time = crawl_time
                    
                    # --- NEW: FETCH CASCADE (COMMENTS) ---
                    permalink = p.get('permalink')
                    cascade_data = []
                    if permalink:
                        cascade_data = self.fetch_comments(permalink)
                    # -------------------------------------

                    media_url = self.classify_media(p.get('url_overridden_by_dest', ""), p)
                    
                    # --- EARLY FILTER: Video-only (downstream chỉ xử lý ảnh tĩnh) ---
                    if media_url and re.search(VIDEO_EXTENSIONS, media_url, re.IGNORECASE):
                        self.stats["skipped_video"] += 1
                        continue
                    
                    # --images-only: bỏ qua bài không có ảnh
                    if self.images_only and not media_url:
                        self.stats["skipped_no_image"] += 1
                        continue
                    
                    item = {
                        "id": post_id,
                        "timestamp": post_time,
                        "label": "Unlabeled",
                        "raw_text": raw_text,
                        "media_url": media_url,
                        "user_id": self.standardize_user(p.get('author', '')),
                        "retweet_count": 0, # Kept for schema compatibility
                        "comment_count": int(p.get('num_comments', 0)),
                        "cascade": cascade_data, # <--- NEW FIELD
                        "metadata": {
                            "source": "reddit",
                            "subreddit": group,
                            "crawl_time": crawl_time,
                            "crawler_version": CRAWLER_VERSION,
                            "run_id": self.run_id
                        }
                    }
                    
                    all_new_items.append(item)
                    existing_ids.add(post_id)
                    group_count += 1
                    self.stats["new"] += 1
                    
                self.logger.info(f"   ✅ Fetched {group_count} new posts from r/{group}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed to crawl r/{group}: {e}")
                self.stats["errors"] += 1
                
            time.sleep(1.5) # Compliance with API limits

        self.save(all_new_items)
        self.logger.info(f"🏁 Run Summary: New: {self.stats['new']} | Skipped empty: {self.stats['skipped_empty']} | Skipped short text: {self.stats['skipped_text_too_short']} | Skipped video: {self.stats['skipped_video']} | Skipped no image: {self.stats['skipped_no_image']} | Errors: {self.stats['errors']}")

    def save(self, items):
        if not items:
            self.logger.info("😴 No new items to save.")
            return
            
        if self.debug:
            self.logger.info(f"🧪 [DEBUG] Would save {len(items)} items to {OUTPUT_FILE}")
            return

        try:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"💾 Successfully saved {len(items)} items to {OUTPUT_FILE}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit Crawler with Best Practices")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (don't save, log verbose)")
    parser.add_argument("--limit", type=int, default=25, help="Number of posts per subreddit (max 100)")
    parser.add_argument("--auto", action="store_true", help="Chạy tự động lặp lại theo chu kỳ")
    parser.add_argument("--interval", type=int, default=15, help="Khoảng cách giữa các lần cào (phút, mặc định: 15)")
    parser.add_argument("--images-only", action="store_true", help="Chỉ lấy bài có hình ảnh (bỏ qua text-only)")
    parser.add_argument("--min-text", type=int, default=10, help="Độ dài tối thiểu text sau khi clean (mặc định: 10 ký tự)")
    parser.add_argument("--subreddits", nargs="+", default=None,
                        help="Danh sách subreddit tùy chỉnh (mặc định: dùng list có sẵn)")
    parser.add_argument("--image-heavy", action="store_true",
                        help="Dùng danh sách subreddit có nhiều ảnh (thay vì list news mặc định)")
    args = parser.parse_args()

    # Chọn danh sách subreddit
    if args.subreddits:
        subreddit_list = args.subreddits
    elif args.image_heavy:
        subreddit_list = SUBREDDITS_IMAGE_HEAVY
    else:
        subreddit_list = SUBREDDITS
    
    print(f"📋 Subreddits: {', '.join(subreddit_list)}")
    if args.images_only:
        print(f"🖼️  Mode: IMAGES ONLY")
    
    crawler = RedditCrawler(
        debug=args.debug, 
        limit=args.limit,
        subreddits=subreddit_list,
        images_only=args.images_only,
        min_text_length=args.min_text
    )

    if args.auto:
        run_count = 0   
        print(f"🔁 Chế độ AUTO: Cào mỗi {args.interval} phút. Nhấn Ctrl+C để dừng.")
        print()
        while True:
            try:
                run_count += 1
                print(f"{'='*60}")
                print(f"📡 LẦN CÀO #{run_count}")
                print(f"{'='*60}")
                crawler = RedditCrawler(
                    debug=args.debug, 
                    limit=args.limit,
                    subreddits=subreddit_list,
                    images_only=args.images_only,
                    min_text_length=args.min_text
                )
                crawler.crawl()
                
                # Đếm ngược
                wait_seconds = args.interval * 60
                print(f"\n⏳ Đợi {args.interval} phút đến lần cào tiếp...")
                for remaining in range(wait_seconds, 0, -60):
                    mins = remaining // 60
                    print(f"   ⏰ Còn {mins} phút...", end='\r')
                    time.sleep(min(60, remaining))
                print()
            except KeyboardInterrupt:
                print(f"\n\n🛑 Đã dừng sau {run_count} lần cào.")
                break
    else:
        crawler.crawl()