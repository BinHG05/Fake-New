import cloudscraper
from bs4 import BeautifulSoup
import json
import time
import os
import re
import logging
import argparse
import uuid
from urllib.parse import urljoin

# --- CONFIGURATION ---
NEWS_VERSION = "1.0.0"
VNEXPRESS_RSS = "https://vnexpress.net/rss/tin-moi-nhat.rss"
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(ROOT_DIR, "data", "01_raw", "news")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
OUTPUT_FILE = os.path.join(DATA_DIR, "news_realtime_data.jsonl")

# --- LOGGING SETUP ---
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, "news_crawl.log"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("NewsCrawler")

class NewsCrawler:
    def __init__(self, limit=20, num_pages=1):
        self.logger = setup_logging()
        self.limit = limit
        self.num_pages = num_pages
        
        # Use cloudscraper to bypass 406 Not Acceptable and Cloudflare
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
            'Referer': 'https://vnexpress.net/',
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

    def get_articles_from_html(self, num_pages=1):
        all_articles = []
        for page in range(1, num_pages + 1):
            url = f"https://vnexpress.net/tin-moi-nhat-p{page}"
            try:
                self.logger.info(f"   📄 Scanning news page {page}: {url}")
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # VnExpress latest news items usually in h3.title-news
                items = soup.select('h3.title-news a')
                page_count = 0
                for a in items:
                    link = a['href']
                    title = a.get_text(strip=True)
                    
                    id_match = re.search(r'-(\d+)\.html', link)
                    if not id_match: continue
                    
                    article_id = "vnexpress_" + id_match.group(1)
                    if article_id in self.existing_ids: continue
                    
                    all_articles.append({
                        "id": article_id,
                        "url": link,
                        "title": title
                    })
                    page_count += 1
                    if len(all_articles) >= self.limit: break
                
                self.logger.info(f"   ✅ Page {page}: Found {page_count} new articles.")
                if len(all_articles) >= self.limit: break
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error fetching HTML page {page}: {e}")
                break
        return all_articles

    def get_articles_from_rss(self):
        try:
            resp = self.session.get(VNEXPRESS_RSS, timeout=15)
            resp.raise_for_status()
            
            # Robust extraction using regex to avoid XML parser issues
            items_raw = re.findall(r'<item>(.*?)</item>', resp.text, re.DOTALL)
            articles = []
            
            for body in items_raw:
                link_match = re.search(r'<link>(.*?)</link>', body)
                title_match = re.search(r'<title>(.*?)</title>', body)
                
                if not link_match or not title_match: continue
                
                link = link_match.group(1).strip()
                title = title_match.group(1).strip()
                # Clean CDATA
                title = title.replace('<![CDATA[', '').replace(']]>', '')
                
                # Extract ID: ...-4712345.html
                id_match = re.search(r'-(\d+)\.html', link)
                if not id_match: continue
                
                article_id = "vnexpress_" + id_match.group(1)
                if article_id in self.existing_ids: continue
                
                articles.append({
                    "id": article_id,
                    "url": link,
                    "title": title
                })
                if len(articles) >= self.limit: break
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching RSS: {e}")
            return []

    def get_comments(self, article_id_raw):
        """
        VnExpress Comment API (Updated 2024/2025):
        https://usi-saas.vnexpress.net/index/get?offset=0&limit=100&fromadmin=0&objectid={article_id}&objecttype=1&siteid=1000000
        """
        try:
            # Domain usi-saas.vnexpress.net is the current working endpoint
            comment_url = f"https://usi-saas.vnexpress.net/index/get?offset=0&limit=100&fromadmin=0&objectid={article_id_raw}&objecttype=1&siteid=1000000"
            resp = self.session.get(comment_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            # The data structure in usi-saas API: data['data']['items']
            raw_comments = data.get('data', {}).get('items', [])
            if not raw_comments:
                return []
                
            cascade = []
            for c in raw_comments:
                c_id = "vnc_" + str(c.get('comment_id'))
                c_text = c.get('content', '')
                c_author = c.get('full_name', 'Anonymous')
                c_time = int(c.get('creation_time', time.time()))
                
                cascade.append({
                    "id": c_id,
                    "parent_id": "vnexpress_" + str(article_id_raw),
                    "user_id": c_author,
                    "timestamp": c_time,
                    "text": c_text
                })
                
                # Check for replies (nested comments)
                # In USI-SAAS API, replies are usually in 'replys' (note the 's')
                replies_data = c.get('replys', {})
                if isinstance(replies_data, dict):
                    replies = replies_data.get('items', [])
                    for r in replies:
                        r_id = "vnc_" + str(r.get('comment_id'))
                        cascade.append({
                            "id": r_id,
                            "parent_id": c_id,
                            "user_id": r.get('full_name', 'Anonymous'),
                            "timestamp": int(r.get('creation_time', time.time())),
                            "text": r.get('content', '')
                        })
            return cascade
        except Exception as e:
            self.logger.warning(f"Could not fetch comments for {article_id_raw}: {e}")
            return []

    def parse_article(self, article_url, article_id, title):
        try:
            resp = self.session.get(article_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Content
            description = soup.select_one('p.description')
            desc_text = description.get_text(strip=True) if description else ""
            
            content_div = soup.select_one('article.fck_detail')
            body_text = content_div.get_text(separator=' ', strip=True) if content_div else ""
            
            # Image
            media_url = ""
            img_tag = soup.select_one('article.fck_detail img')
            if img_tag and ('src' in img_tag.attrs or 'data-src' in img_tag.attrs):
                media_url = img_tag.get('data-src') or img_tag.get('src')

            raw_text = f"{title}. {desc_text}. {body_text}"
            
            # Extract raw ID for comment API
            raw_id = article_id.replace("vnexpress_", "")
            cascade = self.get_comments(raw_id)
            
            return {
                "id": article_id,
                "timestamp": int(time.time()),
                "label": "REAL", # News is typically REAL
                "raw_text": raw_text,
                "media_url": media_url,
                "user_id": "VnExpress",
                "retweet_count": 0,
                "comment_count": len(cascade),
                "cascade": cascade,
                "metadata": {
                    "source": "vnexpress",
                    "url": article_url,
                    "crawler_version": NEWS_VERSION
                }
            }
        except Exception as e:
            self.logger.error(f"Error parsing article {article_url}: {e}")
            return None

    def crawl(self):
        self.logger.info(f"🚀 Starting News Crawl (Run ID: {self.run_id}, Pages: {self.num_pages})")
        # Try HTML pagination first for volume, fallback to RSS for freshness
        articles = self.get_articles_from_html(num_pages=self.num_pages)
        if not articles:
            articles = self.get_articles_from_rss()
        
        self.logger.info(f"Found total {len(articles)} new articles.")
        
        all_items = []
        for a in articles:
            self.logger.info(f"  - Parsing: {a['title'][:50]}...")
            item = self.parse_article(a['url'], a['id'], a['title'])
            if item:
                all_items.append(item)
            time.sleep(1) # Politeness
            
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
    parser = argparse.ArgumentParser(description="Vietnamese News Crawler")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to crawl")
    parser.add_argument("--auto", action="store_true", help="Run in auto mode (every N minutes)")
    parser.add_argument("--interval", type=int, default=30, help="Interval in minutes for auto mode")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to crawl back")
    args = parser.parse_args()
    
    if args.auto:
        print(f"🔁 Chế độ AUTO: Cào mỗi {args.interval} phút. Nhấn Ctrl+C để dừng.")
        while True:
            crawler = NewsCrawler(limit=args.limit, num_pages=args.pages)
            crawler.crawl()
            print(f"💤 Đang đợi {args.interval} phút...")
            time.sleep(args.interval * 60)
    else:
        crawler = NewsCrawler(limit=args.limit, num_pages=args.pages)
        crawler.crawl()
