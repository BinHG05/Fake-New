import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

def inject_features(features_path, graph_dir, output_dir):
    print(f"🚀 Loading features from {features_path}...")
    if not os.path.exists(features_path):
        print(f"❌ Error: Features file not found!")
        return
    
    # Load raw feature dictionary
    raw_data = torch.load(features_path, weights_only=False)
    
    # Pre-process: Map ID to its full 1793-dim feature vector
    # Structure in .pt: ids[N], clip_text_emb[N, 512], clip_image_emb[N, 512], xlmr_emb[N, 768], clip_score[N]
    feat_map = {}
    ids = raw_data["ids"]
    c_text = raw_data["clip_text_emb"]
    c_img = raw_data["clip_image_emb"]
    xlmr = raw_data["xlmr_emb"]
    c_score = raw_data["clip_score"].unsqueeze(1) # [N, 1]
    
    print(f"🔄 Mapping features for {len(ids)} IDs...")
    for i, post_id in enumerate(ids):
        # Build the vector: [XLMR(768), CLIP_T(512), CLIP_I(512), Mismatch(1)] -> Total 1793
        # We put XLMR first to match the GNN script's expectation of 768 text dim
        vector = torch.cat([xlmr[i], c_text[i], c_img[i], c_score[i]], dim=-1)
        feat_map[post_id] = vector
    
    print(f"✅ Created feature map for {len(feat_map)} records.")

    os.makedirs(output_dir, exist_ok=True)
    
    graph_files = list(Path(graph_dir).glob("*.pt"))
    print(f"📂 Processing {len(graph_files)} graphs in {graph_dir}...")

    success_count = 0
    missing_count = 0

    for graph_path in tqdm(graph_files):
        post_id = graph_path.stem
        
        if post_id not in feat_map:
            missing_count += 1
            continue
        
        try:
            # Load graph
            data = torch.load(graph_path, weights_only=False)
            
            # Get the reordered feature vector
            new_feat = feat_map[post_id] 
            
            # Broadcast new feature to all nodes in this graph
            num_nodes = data.num_nodes
            new_x = new_feat.unsqueeze(0).repeat(num_nodes, 1)
            
            # Update graph
            data.x = new_x
            
            # Save to new location
            torch.save(data, os.path.join(output_dir, f"{post_id}.pt"))
            success_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {post_id}: {e}")

    print(f"\n======================================")
    print(f"✅ Injection Complete!")
    print(f"📊 Success: {success_count}")
    print(f"❌ Missing Features: {missing_count}")
    print(f"📂 New graphs saved to: {output_dir}")
    print(f"======================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject CLIP features into graphs")
    parser.add_argument("--features", default="data/04_features/Fakeddit/features_v2.pt")
    parser.add_argument("--graph_dir", default="data/processed_graphs")
    parser.add_argument("--output_dir", default="data/processed_graphs_v3")
    args = parser.parse_args()
    
    inject_features(args.features, args.graph_dir, args.output_dir)
