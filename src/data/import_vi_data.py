import json
import os
import logging
from datetime import datetime

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_NEWS_FILE = os.path.join(ROOT_DIR, "data", "01_raw", "news", "news_realtime_data.jsonl")
RAW_VOZ_FILE = os.path.join(ROOT_DIR, "data", "01_raw", "voz", "voz_realtime_data.jsonl")
OUTPUT_MASTER = os.path.join(ROOT_DIR, "data", "vi_master_labeled.jsonl")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImportVIData")

def import_news():
    items = []
    if not os.path.exists(RAW_NEWS_FILE):
        logger.warning(f"File không tồn tại: {RAW_NEWS_FILE}")
        return items
    
    logger.info(f"Đang nhập dữ liệu từ VnExpress...")
    with open(RAW_NEWS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Đảm bảo nhãn là REAL cho báo chí chính thống
                data['label'] = "REAL"
                # Thêm flag nguồn gốc
                data['metadata']['lang'] = 'vi'
                items.append(data)
            except: continue
    logger.info(f"Đã nhập {len(items)} mẫu từ VnExpress (Nhãn: REAL)")
    return items

def import_voz(force_label=None):
    items = []
    if not os.path.exists(RAW_VOZ_FILE):
        logger.warning(f"File không tồn tại: {RAW_VOZ_FILE}")
        return items
    
    logger.info(f"Đang nhập dữ liệu từ Voz Forum...")
    with open(RAW_VOZ_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Nếu người dùng muốn ép nhãn (ví dụ FAKE cho các thread tin đồn)
                if force_label:
                    data['label'] = force_label
                elif data.get('label') == "Unlabeled":
                    data['label'] = "FAKE" # Tạm thời giả định tin từ forum/rumor là FAKE để cân bằng
                
                data['metadata']['lang'] = 'vi'
                items.append(data)
            except: continue
    logger.info(f"Đã nhập {len(items)} mẫu từ Voz (Nhãn: {force_label or 'FAKE (Default)'})")
    return items

def main():
    # 1. Thu thập dữ liệu
    news_data = import_news()
    voz_data = import_voz() # Mặc định gán FAKE để đối trọng với VnExpress (REAL)
    
    all_data = news_data + voz_data
    
    if not all_data:
        logger.error("Không tìm thấy dữ liệu nào để nhập!")
        return

    # 2. Ghi ra file Master chung cho dữ liệu Việt Nam
    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    with open(OUTPUT_MASTER, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    logger.info("=" * 40)
    logger.info(f"HOÀN THÀNH: Đã tạo file Master Tiếng Việt tại:")
    logger.info(f"📍 {OUTPUT_MASTER}")
    logger.info(f"📊 Tổng cộng: {len(all_data)} mẫu (REAL: {len(news_data)}, FAKE: {len(voz_data)})")
    logger.info("=" * 40)
    logger.info("👉 Bước tiếp theo: Chạy trích xuất đặc trưng (CLIP/XLM-R) cho file này.")

if __name__ == "__main__":
    main()
