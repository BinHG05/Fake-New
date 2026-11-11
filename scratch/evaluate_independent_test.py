
import os
import sys
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.train_multimodal_gnn import load_cascade_graphs, MultimodalCascadeGNN, evaluate
from src.deployment.predictor import DEFAULT_MULTIMODAL_CHECKPOINT, TEXT_DIM, IMAGE_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_independent():
    graph_dir = PROJECT_ROOT / "data" / "independent_test" / "graphs"
    metadata_path = PROJECT_ROOT / "data" / "independent_test" / "metadata_final.jsonl"
    checkpoint_path = DEFAULT_MULTIMODAL_CHECKPOINT
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the processed graphs
    # We use a custom loader because we want to load all of them as 'test'
    print("Loading independent test graphs...")
    test_graphs = []
    
    # Load metadata to get labels
    id_to_label = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            id_to_label[item['id']] = item['label']
            
    for pt_file in graph_dir.glob('*.pt'):
        post_id = pt_file.stem
        if post_id in id_to_label:
            data = torch.load(pt_file, weights_only=False)
            data.y = torch.tensor([id_to_label[post_id]], dtype=torch.long)
            test_graphs.append(data)
            
    print(f"Loaded {len(test_graphs)} graphs for evaluation.")
    test_loader = DataLoader(test_graphs, batch_size=32)
    
    # 2. Initialize Model (Adjusted to match checkpoint: fusion_dim=512, hidden_dim=512)
    input_dim = test_graphs[0].x.size(1)
    text_dim = 768
    image_dim = input_dim - text_dim
    
    model = MultimodalCascadeGNN(
        input_dim=input_dim,
        text_dim=text_dim,
        image_dim=image_dim,
        fusion_dim=512,
        hidden_dim=512,
        num_classes=2,
        gnn_type='gat'
    ).to(device)
    
    # 3. Load Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle possible state_dict mismatches (e.g. if checkpoint is only for GNN)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Run Evaluation
    print("\n" + "="*50)
    print("INDEPENDENT TEST RESULTS (1000 SAMPLES)")
    print("="*50)
    
    results = evaluate(model, test_loader, device, split='test')
    
    print(f"Accuracy: {results['test_acc_bin']:.4f}")
    print(f"F1 Score: {results['test_f1_bin']:.4f}")
    print("="*50)

if __name__ == "__main__":
    import json
    evaluate_independent()
