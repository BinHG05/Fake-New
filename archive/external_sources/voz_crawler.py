import cloudscraper
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging
import argparse
import uuid
from datetime import datetime
from urllib.parse import urljoin

# --- CONFIGURATION ---
VOZ_VERSION = "1.0.0"
DEFAULT_FORUMS = {
    "f33": "https://voz.vn/f/diem-bao.33/",
    "f17": "https://voz.vn/f/chuyen-tro-linh-tinh.17/"
}

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "data", "01_raw", "voz")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
OUTPUT_FILE = os.path.join(DATA_DIR, "voz_realtime_data.jsonl")

# --- LOGGING SETUP ---
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "voz_crawl.log"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("VozCrawler")

class VozCrawler:
    def __init__(self, limit=10, forums=None, images_only=False, num_pages=1):
        self.logger = setup_logging()
        self.limit = limit
        self.forums = forums or DEFAULT_FORUMS
        self.images_only = images_only
        self.num_pages = num_pages
        
        # Use cloudscraper to bypass Cloudflare 403
        self.session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'vi,en-US;q=0.9,en;q=0.8',
            'Referer': 'https://voz.vn/',
            'Upgrade-Insecure-Requests': '1',
            'Connection': 'keep-alive',
        })
        self.run_id = str(uuid.uuid4())[:8]
        self.existing_ids = self._get_existing_ids()

    def _get_existing_ids(self):
        ids = set()
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        ids.add(data['id'])
                    except: continue
        return ids

    def get_thread_list(self, forum_url, num_pages=1):
        all_threads = []
        for page in range(1, num_pages + 1):
            page_url = forum_url if page == 1 else f"{forum_url}page-{page}"
            try:
                self.logger.info(f"   📄 Scanning page {page}: {page_url}")
                resp = self.session.get(page_url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # XenForo thread list items
                items = soup.select('.structItem--thread')
                page_threads_count = 0
                for item in items:
                    title_tag = item.select_one('.structItem-title a[data-tp-primary]')
                    if not title_tag: continue
                    
                    thread_url = urljoin("https://voz.vn", title_tag['href'])
                    thread_id = re.search(r'\.(\d+)/$', thread_url)
                    if thread_id:
                        thread_id = "voz_" + thread_id.group(1)
                    else:
                        continue
                    
                    if thread_id in self.existing_ids:
                        continue
                    
                    all_threads.append({
                        "id": thread_id,
                        "url": thread_url,
                        "title": title_tag.get_text(strip=True)
                    })
                    page_threads_count += 1
                    if len(all_threads) >= self.limit: break
                
                self.logger.info(f"   ✅ Page {page}: Found {page_threads_count} new threads.")
                if len(all_threads) >= self.limit: break
                time.sleep(1) # Small delay between pages
                
            except Exception as e:
                self.logger.error(f"Error fetching page {page} from {forum_url}: {e}")
                break
        return all_threads

    def parse_thread(self, thread_url, thread_id, title):
        try:
            resp = self.session.get(thread_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 1. Get Root Post (First post in thread)
            first_post = soup.select_one('.message-inner')
            if not first_post: return None
            
            author = first_post.select_one('.message-name').get_text(strip=True)
            content_tag = first_post.select_one('.bbWrapper')
            
            # Extract real image (skip SVG placeholders/emoticons)
            media_url = ""
            all_imgs = content_tag.find_all('img')
            for img in all_imgs:
                src = img.get('src', '') or img.get('data-src', '')
                if not src: continue
                
                # Skip data URIs (SVGs/Placeholders)
                if src.startswith('data:'): continue
                
                # Skip common emoticons/icons
                if 'styles/default/xenforo/smilies' in src: continue
                
                # Resolve URL
                full_url = urljoin("https://voz.vn", src)
                
                # Prefer actual news images (usually http or internal attachments)
                media_url = full_url
                break
            
            if self.images_only and not media_url:
                return None

            raw_text = f"{title} {content_tag.get_text(separator=' ', strip=True)}"
            
            # 2. Get Comments (Cascade)
            # For simplicity, we only crawl the first page of comments in this version
            cascade = []
            all_posts = soup.select('.message--post')[1:] # Skip the first post (root)
            
            for post in all_posts:
                p_id_match = re.search(r'post-(\d+)', post.get('id', ''))
                if not p_id_match: continue
                p_id = "vp_" + p_id_match.group(1)
                
                p_author = post.select_one('.message-name').get_text(strip=True)
                p_content = post.select_one('.bbWrapper')
                
                # Check for quotes to build tree
                parent_id = thread_id # Default parent is the thread
                quote = p_content.select_one('.bbCodeBlock--quote')
                if quote:
                    # Try to find who was quoted
                    quote_attr = quote.get('data-quote') # e.g. "User Name, post: 123456"
                    if quote_attr:
                        quoted_post_id_match = re.search(r'post:\s*(\d+)', quote_attr)
                        if quoted_post_id_match:
                            parent_id = "vp_" + quoted_post_id_match.group(1)
                    
                    # Remove quote text from body to get actual reply
                    quote.decompose()
                
                p_text = p_content.get_text(separator=' ', strip=True)
                if not p_text: continue
                
                p_time_tag = post.select_one('time.u-dt')
                try:
                    if p_time_tag and 'data-time' in p_time_tag.attrs:
                        p_time = int(p_time_tag['data-time'])
                    else:
                        p_time = int(time.time())
                except:
                    p_time = int(time.time())
                
                cascade.append({
                    "id": p_id,
                    "parent_id": parent_id,
                    "user_id": p_author,
                    "timestamp": p_time,
                    "text": p_text
                })

            return {
                "id": thread_id,
                "timestamp": int(time.time()),
                "label": "Unlabeled",
                "raw_text": raw_text,
                "media_url": media_url,
                "user_id": author,
                "retweet_count": 0,
                "comment_count": len(cascade),
                "cascade": cascade,
                "metadata": {
                    "source": "voz",
                    "url": thread_url,
                    "crawler_version": VOZ_VERSION
                }
            }
        except Exception as e:
            self.logger.error(f"Error parsing thread {thread_url}: {e}")
            return None

    def crawl(self):
        self.logger.info(f"🚀 Starting Voz Crawl (Run ID: {self.run_id}, Pages: {self.num_pages})")
        all_items = []
        
        for name, url in self.forums.items():
            self.logger.info(f"📡 Scanning forum: {name}...")
            threads = self.get_thread_list(url, num_pages=self.num_pages)
            self.logger.info(f"Found {len(threads)} new threads.")
            
            for t in threads:
                self.logger.info(f"  - Parsing: {t['title'][:50]}...")
                item = self.parse_thread(t['url'], t['id'], t['title'])
                if item:
                    all_items.append(item)
                time.sleep(2) # Be polite
                
        self.save(all_items)

    def save(self, items):
        if not items:
            self.logger.info("No new items saved.")
            return
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.logger.info(f"💾 Saved {len(items)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voz.vn Crawler")
    parser.add_argument("--limit", type=int, default=10, help="Threads per forum")
    parser.add_argument("--images-only", action="store_true", help="Only save threads with images")
    parser.add_argument("--auto", action="store_true", help="Run in auto mode (every N minutes)")
    parser.add_argument("--interval", type=int, default=15, help="Interval in minutes for auto mode")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to crawl per forum")
    args = parser.parse_args()
    
    if args.auto:
        print(f"🔁 Chế độ AUTO (Voz): Cào mỗi {args.interval} phút. Nhấn Ctrl+C để dừng.")
        while True:
            crawler = VozCrawler(limit=args.limit, images_only=args.images_only, num_pages=args.pages)
            crawler.crawl()
            print(f"💤 Đang đợi {args.interval} phút...")
            time.sleep(args.interval * 60)
    else:
        crawler = VozCrawler(limit=args.limit, images_only=args.images_only, num_pages=args.pages)
        crawler.crawl()
