"""
Rebuild Cascade Graphs with Image Features (ResNet50).

Adds ResNet50 image embeddings to the root node of each cascade graph.
Node features go from [N, 768] (text only) to [N, 768+512] = [N, 1280].

Uses torchvision ResNet50 (no HuggingFace dependency, works with any PyTorch version).

Usage:
    python src/utils/rebuild_graphs_with_images.py
    python src/utils/rebuild_graphs_with_images.py --image_dirs data/02_processed/images/Fakeddit_600_800
"""

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target image embedding dimension (project ResNet 2048 -> 512)
IMAGE_DIM = 512


class ResNetFeatureExtractor:
    """Extract image features using ResNet50 (torchvision, no HF needed)."""

    def __init__(self, device='cuda'):
        self.device = device
        # Load pretrained ResNet50, remove final classification layer
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 2048, 1, 1]
        self.proj = nn.Linear(2048, IMAGE_DIM)  # project to 512d

        # Initialize projection with Xavier
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.backbone = self.backbone.to(device).eval()
        self.proj = self.proj.to(device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"ResNet50 feature extractor loaded on {device} (output: {IMAGE_DIM}d)")

    @torch.no_grad()
    def extract_batch(self, image_paths, batch_size=32):
        """Extract features for a list of image paths. Returns dict: path -> tensor."""
        results = {}
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    images.append(self.transform(img))
                    valid_paths.append(p)
                except Exception as e:
                    logger.debug(f"Skip {p}: {e}")
                    results[p] = torch.zeros(IMAGE_DIM)

            if images:
                batch_tensor = torch.stack(images).to(self.device)
                feats = self.backbone(batch_tensor).squeeze(-1).squeeze(-1)  # [B, 2048]
                feats = self.proj(feats)  # [B, 512]
                feats = F.normalize(feats, p=2, dim=-1)  # L2 normalize

                for j, p in enumerate(valid_paths):
                    results[p] = feats[j].cpu()

        return results


def build_image_index(image_dirs):
    """Build lookup: post_id -> absolute image path."""
    index = {}
    for d in image_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for img in d.iterdir():
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                index[img.stem] = str(img.resolve())
    logger.info(f"Image index: {len(index)} images across {len(image_dirs)} dirs")
    return index


def main():
    parser = argparse.ArgumentParser(description='Rebuild cascade graphs with image features')
    parser.add_argument('--metadata', default='data/reddit_enriched_data.jsonl')
    parser.add_argument('--graph_dir', default='data/processed_graphs')
    parser.add_argument('--output_dir', default='data/processed_graphs_multimodal')
    parser.add_argument('--image_dirs', nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Auto-discover image directories
    if args.image_dirs is None:
        img_base = Path('data/02_processed/images')
        args.image_dirs = [str(d) for d in img_base.iterdir() if d.is_dir()]
        logger.info(f"Auto-discovered {len(args.image_dirs)} image directories")

    # Build image index
    image_index = build_image_index(args.image_dirs)

    # Load metadata
    logger.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, 'r', encoding='utf-8') as f:
        id_to_meta = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            id_to_meta[item['id']] = item

    # Match post IDs to images
    matched_ids = [pid for pid in id_to_meta if pid in image_index]
    unmatched_ids = [pid for pid in id_to_meta if pid not in image_index]
    logger.info(f"Posts with images on disk: {len(matched_ids)} / {len(id_to_meta)}")
    logger.info(f"Posts without images: {len(unmatched_ids)} (will use zero padding)")

    # Extract image features
    logger.info("Loading ResNet50...")
    extractor = ResNetFeatureExtractor(device=device)

    image_paths = [image_index[pid] for pid in matched_ids]
    logger.info(f"Extracting features for {len(image_paths)} images...")

    all_features = {}
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Extracting"):
        batch = image_paths[i:i + args.batch_size]
        batch_results = extractor.extract_batch(batch, batch_size=args.batch_size)
        # Map back path -> post_id
        for path, feat in batch_results.items():
            pid = Path(path).stem
            all_features[pid] = feat

    logger.info(f"Extracted {len(all_features)} image features")

    # Rebuild graphs
    torch.serialization.add_safe_globals([Data])
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)

    processed, with_img, without_img = 0, 0, 0
    for pt_file in tqdm(sorted(graph_dir.glob('*.pt')), desc="Rebuilding graphs"):
        post_id = pt_file.stem
        try:
            data = torch.load(pt_file, weights_only=False)
            num_nodes = data.x.size(0)

            # Image features: root node gets real embedding, rest get zeros
            img_feat = torch.zeros(num_nodes, IMAGE_DIM)
            if post_id in all_features:
                img_feat[0] = all_features[post_id]
                with_img += 1
            else:
                without_img += 1

            # Concat: [text_768 | image_512] = [N, 1280]
            new_x = torch.cat([data.x, img_feat], dim=-1)

            new_data = Data(x=new_x, edge_index=data.edge_index)
            new_data.post_id = post_id
            new_data.num_nodes = num_nodes

            torch.save(new_data, output_dir / f"{post_id}.pt")
            processed += 1

        except Exception as e:
            logger.warning(f"Failed: {pt_file.name}: {e}")

    print(f"\n{'='*50}")
    print(f"REBUILD COMPLETE")
    print(f"{'='*50}")
    print(f"Total graphs:      {processed}")
    print(f"With real images:   {with_img} ({100*with_img/max(processed,1):.1f}%)")
    print(f"Without images:     {without_img} (zero-padded)")
    print(f"Feature dim:        768 (text) + {IMAGE_DIM} (image) = {768+IMAGE_DIM}")
    print(f"Output:             {output_dir}")
    print(f"{'='*50}")
    print(f"\nNext step:")
    print(f"  python src/training/train_multimodal_gnn.py --graph_dir {output_dir} --epochs 100 --lr 1e-3 --patience 15")


if __name__ == '__main__':
    main()
