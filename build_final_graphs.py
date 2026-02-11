import torch
import json
import os
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
    for data in graphs:
        save_path = os.path.join(OUTPUT_FOLDER, f"{data.post_id}.pt")
        torch.save(data, save_path)
        print(f"Đã tạo đồ thị cho bài: {data.post_id}")

if __name__ == "__main__":
    main()