
import os
import json
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.cascade_graph_builder import CascadeGraphBuilder
from src.deployment.predictor import CLIPFeatureExtractor
from src.data.label_utils import derive_binary_label, BINARY_LABEL_MAP

def process_independent_test():
    input_file = PROJECT_ROOT / "data" / "independent_test" / "raw_metadata.jsonl"
    output_dir = PROJECT_ROOT / "data" / "independent_test" / "graphs"
    output_metadata = PROJECT_ROOT / "data" / "independent_test" / "metadata_final.jsonl"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize components
    graph_builder = CascadeGraphBuilder(device=device)
    clip_extractor = CLIPFeatureExtractor(device=device)
    
    processed_count = 0
    metadata_list = []
    
    print(f"Processing 1000 samples from {input_file}...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        try:
            item = json.loads(line.strip())
            post_id = item["id"]
            
            # 1. Build Graph Structure + Text Embeddings
            # Cascade is empty in this file, so it will be a single-node graph
            graph = graph_builder.build_graph(item)
            if graph is None:
                continue
                
            # 2. Extract CLIP Features (Title + Image)
            # Fakeddit usually has media_url. Predictor logic will try to download or find local.
            # For this independent test, we'll use the title for CLIP text
            title = item.get("raw_text", "").split("\n")[0]
            
            # Note: We don't have local images for these new samples usually, 
            # so it will fallback to zero-image features or try downloading if media_url exists.
            # In a research context, if we want full multimodal, we need the images.
            # For now, we follow the project's logic.
            
            # Mocking the image path as None to use text-only CLIP or try download 
            # (Predictor has download logic, but let's keep it simple here)
            image_vector = clip_extractor.extract(title, image_path=None)
            
            # 3. Concatenate Features (1793 dims)
            # graph.x is [N, 768]
            num_nodes = graph.x.size(0)
            image_features = torch.zeros((num_nodes, 1025), dtype=graph.x.dtype)
            image_features[0] = image_vector.to(dtype=graph.x.dtype)
            
            graph.x = torch.cat([graph.x, image_features], dim=-1)
            
            # 4. Save Graph
            output_path = output_dir / f"{post_id}.pt"
            torch.save(graph, output_path)
            
            # 5. Update Metadata
            # Label conversion
            try:
                binary_label, _ = derive_binary_label(item)
                label_idx = BINARY_LABEL_MAP[binary_label]
            except:
                label_idx = 1 if item.get("label", "").lower() == "fake" else 0
                
            metadata_item = {
                "id": post_id,
                "label": label_idx,
                "split": "test", # Mark as test for the training script
                "raw_text": item.get("raw_text", ""),
                "label_binary": "FAKE" if label_idx == 1 else "REAL"
            }
            metadata_list.append(metadata_item)
            processed_count += 1
            
        except Exception as e:
            # print(f"Error processing {line[:50]}...: {e}")
            continue

    # Save final metadata
    with open(output_metadata, "w", encoding="utf-8") as f:
        for entry in metadata_list:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully processed {processed_count} samples.")
    print(f"Graphs saved in: {output_dir}")
    print(f"Metadata saved in: {output_metadata}")

if __name__ == "__main__":
    process_independent_test()
