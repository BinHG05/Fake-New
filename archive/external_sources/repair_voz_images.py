import json
import os
import time
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

# --- CONFIG ---
RAW_FILE = "data/01_raw/voz/voz_realtime_data.jsonl"
TEMP_FILE = "data/01_raw/voz/voz_realtime_data_fixed.jsonl"

def get_real_image(url, scraper):
    try:
        resp = scraper.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        first_post = soup.select_one('.message-inner')
        if not first_post: return None
        
        content_tag = first_post.select_one('.bbWrapper')
        if not content_tag: return None
        
        # New logic: find first real image
        all_imgs = content_tag.find_all('img')
        for img in all_imgs:
            src = img.get('src', '') or img.get('data-src', '')
            if not src or src.startswith('data:') or 'styles/default' in src:
                continue
            return urljoin("https://voz.vn", src)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def main():
    if not os.path.exists(RAW_FILE):
        print(f"File not found: {RAW_FILE}")
        return

    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )

    print("Reading records...")
    records = []
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Total records: {len(records)}")
    
    # Process
    fixed_count = 0
    skipped_count = 0
    
    with open(TEMP_FILE, 'w', encoding='utf-8') as f_out:
        for record in tqdm(records, desc="Repairing Images"):
            url = record.get('metadata', {}).get('url')
            current_media = record.get('media_url', '')
            
            # If current media is a data URI or empty, try to repair
            if url and (current_media.startswith('data:') or not current_media):
                new_img = get_real_image(url, scraper)
                if new_img:
                    record['media_url'] = new_img
                    fixed_count += 1
                else:
                    skipped_count += 1
                time.sleep(1.5) # Polite delay
            else:
                skipped_count += 1
            
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Replace old file
    os.replace(TEMP_FILE, RAW_FILE)
    print(f"\nDONE!")
    print(f"Fixed: {fixed_count} records")
    print(f"Skipped: {skipped_count} records")
    print(f"Updated file saved to {RAW_FILE}")

if __name__ == "__main__":
    main()
