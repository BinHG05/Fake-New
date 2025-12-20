"""
Graph Preprocessor CLI - Build interaction graph from labeled JSONL.

Usage:
    python src/data/preprocessor_graph.py --input merged_data.jsonl --output graph.pt
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def build_graph_from_jsonl(
    input_path: str,
    output_path: str,
    text_model: str = 'xlm-roberta-base',
    image_model: str = 'openai/clip-vit-base-patch32',
    k_text: int = 5,
    k_image: int = 5,
    mode: str = 'prototype'
) -> None:
    """
    Build interaction graph from merged JSONL file.
    
    Args:
        input_path: Path to merged JSONL
        output_path: Path to output .pt file
        text_model: HuggingFace text model name
        image_model: HuggingFace CLIP model name
        k_text: Top-K text neighbors
        k_image: Top-K image neighbors
        mode: 'prototype' or 'scale'
    """
    from src.features.embedding_extractor import TextEmbeddingExtractor, ImageEmbeddingExtractor
    from src.features.graph_builder import InteractionGraphBuilder
    
    print("=" * 60)
    print("INTERACTION GRAPH BUILDER")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Text model: {text_model}")
    print(f"Image model: {image_model}")
    print(f"K_text: {k_text}, K_image: {k_image}")
    print(f"Mode: {mode}")
    print("=" * 60)
    print()
    
    # Create extractors
    print("🚀 Initializing embedding extractors...")
    text_ext = TextEmbeddingExtractor(model_name=text_model)
    image_ext = ImageEmbeddingExtractor(model_name=image_model)
    
    # Create builder
    builder = InteractionGraphBuilder(
        text_extractor=text_ext,
        image_extractor=image_ext,
        k_text=k_text,
        k_image=k_image,
        mode=mode
    )
    
    # Build graph
    print()
    graph = builder.build_graph(input_path, project_root=project_root)
    
    # Save
    print()
    builder.save_graph(graph, output_path)
    
    # Summary
    print()
    print("=" * 60)
    print("✅ GRAPH BUILD COMPLETE")
    print("=" * 60)
    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {graph.num_edges}")
    print(f"Node features: {graph.x.shape}")
    print(f"Labels (6-class): {graph.y.unique().tolist()}")
    print(f"Labels (binary): {graph.y_binary.unique().tolist()}")
    print(f"Train: {graph.train_mask.sum().item()}")
    print(f"Val:   {graph.val_mask.sum().item()}")
    print(f"Test:  {graph.test_mask.sum().item()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Build interaction graph from labeled JSONL'
    )
    parser.add_argument(
        '--input',
        default='data/04_graph/merged_data.jsonl',
        help='Path to merged JSONL file'
    )
    parser.add_argument(
        '--output',
        default='data/04_graph/fakeddit_graph.pt',
        help='Path to output graph .pt file'
    )
    parser.add_argument(
        '--text-model',
        default='xlm-roberta-base',
        help='Text embedding model (default: xlm-roberta-base)'
    )
    parser.add_argument(
        '--image-model',
        default='openai/clip-vit-base-patch32',
        help='Image embedding model (default: openai/clip-vit-base-patch32)'
    )
    parser.add_argument(
        '--k-text',
        type=int,
        default=5,
        help='Top-K text neighbors (default: 5)'
    )
    parser.add_argument(
        '--k-image',
        type=int,
        default=5,
        help='Top-K image neighbors (default: 5)'
    )
    parser.add_argument(
        '--mode',
        choices=['prototype', 'scale'],
        default='prototype',
        help='Mode: prototype (torch) or scale (FAISS)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    build_graph_from_jsonl(
        input_path=args.input,
        output_path=args.output,
        text_model=args.text_model,
        image_model=args.image_model,
        k_text=args.k_text,
        k_image=args.k_image,
        mode=args.mode
    )


if __name__ == "__main__":
    main()