
import torch
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import os
from pathlib import Path

def visualize_graph(graph_path, output_html='data/graph_visualization.html'):
    # Load graph
    print(f"📥 Loading graph from: {graph_path}")
    data = torch.load(graph_path, weights_only=False)
    
    # Chuyển đổi sang NetworkX
    G = nx.Graph()
    
    # Định nghĩa màu sắc cho các nhãn (6-class)
    # 0: TRUE, 1: MOSTLY_TRUE, 2: HALF_TRUE, 3: BARELY_TRUE, 4: FALSE, 5: PANTS_ON_FIRE
    label_colors = {
        0: "#2ecc71", # Xanh lá (True)
        1: "#27ae60", # Xanh đậm hơn
        2: "#f39c12", # Cam (Half True)
        3: "#e67e22", # Cam đậm
        4: "#e74c3c", # Đỏ (False)
        5: "#c0392b"  # Đỏ đậm (Pants on Fire)
    }
    
    label_names = {
        0: "TRUE", 1: "MOSTLY_TRUE", 2: "HALF_TRUE", 
        3: "BARELY_TRUE", 4: "FALSE", 5: "PANTS_ON_FIRE"
    }

    print("🔄 Building interactive visualization...")
    
    # Add nodes
    for i in range(data.num_nodes):
        node_label = int(data.y[i])
        color = label_colors.get(node_label, "#95a5a6")
        split = "Train" if data.train_mask[i] else ("Val" if data.val_mask[i] else "Test")
        
        G.add_node(i, 
                   label=f"Post {i}", 
                   title=f"ID: {i}\nLabel: {label_names[node_label]}\nSplit: {split}",
                   color=color,
                   size=20)

    # Add edges
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        e_type = int(edge_attr[i])
        
        # 0: Text similarity (Xanh dương), 1: Image similarity (Tím)
        color = "#3498db" if e_type == 0 else "#9b59b6"
        label = "Text Similarity" if e_type == 0 else "Image Similarity"
        
        G.add_edge(src, dst, color=color, title=label, width=1)

    # Tạo bản đồ tương tác với PyVis
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=False)
    net.from_nx(G)
    
    # Cấu hình vật lý để nhìn cho đẹp (Physics)
    net.toggle_physics(True)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)
    
    # Lưu file
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    
    print(f"✅ Đã tạo xong! Bạn hãy mở file này bằng trình duyệt: \n👉 {os.path.abspath(output_html)}")

if __name__ == "__main__":
    visualize_graph('data/04_graph/fakeddit_graph.pt')
