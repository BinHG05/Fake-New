import torch
import json
import os
import sys

# Đảm bảo thư mục gốc project nằm trong sys.path
# (src/utils/build_final_graphs.py -> lùi 2 cấp = project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.cascade_graph_builder import CascadeGraphBuilder

def main():
    # Khai báo đầu vào và đầu ra
    INPUT_FILE = "data/reddit_enriched_data.jsonl"
    OUTPUT_FOLDER = "data/processed_graphs"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Khởi tạo Builder (Nó sẽ tự động gọi embedding_extractor bên trong)
    # Lưu ý: Lần đầu chạy sẽ tốn thời gian tải model XLM-RoBERTa về máy
    builder = CascadeGraphBuilder()

    # Đọc dữ liệu đã làm giàu
    items = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))

    # Chạy dây chuyền xử lý hàng loạt
    # Hàm này biến text -> vector số (x) và nối comment -> cạnh (edge_index)
    graphs = builder.process_dataset(items)

    # Lưu kết quả thành từng file .pt
    # Lưu kết quả thành từng file .pt
    for data in graphs:
        save_path = os.path.join(OUTPUT_FOLDER, f"{data.post_id}.pt")
        
        # Kiểm tra nếu file đã tồn tại thì bỏ qua (Resume)
        if os.path.exists(save_path):
            print(f"⏩ [SKIP] Đồ thị đã tồn tại: {data.post_id}")
            continue

        torch.save(data, save_path)
        print(f"✅ [NEW] Đã tạo đồ thị cho bài: {data.post_id}")

if __name__ == "__main__":
    main()