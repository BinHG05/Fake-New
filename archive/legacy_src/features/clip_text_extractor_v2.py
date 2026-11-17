"""
Feature Extractor v2: CLIP + XLM-RoBERTa embeddings.

Reads labeled_master_binary.jsonl and extracts:
  - CLIP text embedding (512d)
  - CLIP image embedding (512d)
  - CLIP text-image mismatch score (1d)
  - XLM-RoBERTa text embedding (768d)
  - Old text features (3d) for v1 baseline comparison

Saves everything in a single .pt file for downstream training.

Usage:
    python src/features/clip_text_extractor_v2.py
"""

import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # d:\NCKH_Project\Project


def resolve_image_path(raw_path: str) -> Path:
    """
    Convert Label Studio / legacy paths to an actual local filesystem path.

    Examples of raw_path formats in the dataset:
      - "/data/local-files/?d=02_processed/images/Fakeddit_20251212_180215/9g2b4h.jpg"
      - "/data/local-files/?d=data/02_processed/images/Fakeddit_600_800/xxx.jpg"
      - "data\\02_processed\\images\\Reddit_reddit_run_009\\1rj9f0y.jpg"
    """
    p = raw_path.replace("\\", "/")

    # Strip Label Studio prefix
    if "?d=" in p:
        p = p.split("?d=")[-1]

    # Remove leading slash or "data/" duplication
    p = p.lstrip("/")
    if p.startswith("data/"):
        # Already relative to project root
        return PROJECT_ROOT / p
    else:
        # Assume it's relative to data/
        return PROJECT_ROOT / "data" / p


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset(jsonl_path: str) -> List[Dict]:
    """Load the labeled_master_binary.jsonl file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_features(
    records: List[Dict],
    clip_model_name: str = "openai/clip-vit-base-patch32",
    xlmr_model_name: str = "xlm-roberta-base",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Extract CLIP + XLM-R features for all records.

    Returns a dict ready to be saved with torch.save().
    """
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = len(records)
    print(f"📦 Total records: {N}")
    print(f"🖥️  Device: {device}")

    # ── Load models ────────────────────────────────────────────────────────
    print(f"\n📥 Loading CLIP model: {clip_model_name} ...")
    clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_name)
    print("   ✅ CLIP loaded")

    print(f"📥 Loading XLM-R model: {xlmr_model_name} ...")
    xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
    xlmr_model = AutoModel.from_pretrained(xlmr_model_name, use_safetensors=True).to(device).eval()
    print("   ✅ XLM-R loaded")

    # ── Prepare storage ────────────────────────────────────────────────────
    ids = []
    labels = []
    splits = []
    clip_text_embs = []
    clip_image_embs = []
    xlmr_embs = []
    clip_scores = []
    old_text_feats = []
    image_found_flags = []

    # ── Process in batches ─────────────────────────────────────────────────
    t0 = time.time()
    n_img_found = 0
    n_img_missing = 0

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_records = records[batch_start:batch_end]
        B = len(batch_records)

        # --- Gather texts and image paths ---
        texts = []
        img_paths = []
        for rec in batch_records:
            ids.append(rec["id"])
            labels.append(0 if rec.get("label_binary", rec.get("label")) == "REAL" else 1)
            splits.append(rec.get("split", "train"))

            texts.append(rec.get("clean_text", "") or "")

            img_info = rec.get("image_info", {})
            raw_p = img_info.get("processed_path", "")
            img_paths.append(resolve_image_path(raw_p))

            # Old v1 features
            tf = rec.get("text_features", {})
            old_text_feats.append([
                float(tf.get("word_count", 0)),
                float(tf.get("has_caps_lock", False)),
                float(tf.get("sentiment_score", 0.0)),
            ])

        # ── 1) CLIP text embeddings (512d) ──
        clip_text_inputs = clip_tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        # Call sub-modules directly (compatible with all transformers versions)
        text_outputs = clip_model.text_model(**clip_text_inputs)
        clip_text_out = clip_model.text_projection(text_outputs.pooler_output)  # [B, 512]
        clip_text_out = F.normalize(clip_text_out, p=2, dim=-1).cpu()
        clip_text_embs.append(clip_text_out)

        # ── 2) CLIP image embeddings (512d) ──
        batch_img_embs = torch.zeros(B, 512)
        batch_img_found = [False] * B
        valid_images = []
        valid_indices = []

        for i, img_p in enumerate(img_paths):
            if img_p.exists():
                try:
                    img = Image.open(img_p).convert("RGB")
                    valid_images.append(img)
                    valid_indices.append(i)
                except Exception:
                    pass

        if valid_images:
            pixel_inputs = clip_processor(
                images=valid_images, return_tensors="pt"
            )["pixel_values"].to(device)
            # Call sub-modules directly
            vision_outputs = clip_model.vision_model(pixel_values=pixel_inputs)
            img_out = clip_model.visual_projection(vision_outputs.pooler_output)  # [V, 512]
            img_out = F.normalize(img_out, p=2, dim=-1).cpu()
            for j, orig_idx in enumerate(valid_indices):
                batch_img_embs[orig_idx] = img_out[j]
                batch_img_found[orig_idx] = True


        clip_image_embs.append(batch_img_embs)
        for found in batch_img_found:
            image_found_flags.append(found)
            if found:
                n_img_found += 1
            else:
                n_img_missing += 1

        # ── 3) CLIP mismatch score ──
        # Cosine similarity between text and image embeddings for each sample
        batch_scores = (clip_text_out * batch_img_embs).sum(dim=-1)  # [B]
        clip_scores.append(batch_scores)

        # ── 4) XLM-R text embeddings (768d) ──
        xlmr_inputs = xlmr_tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        xlmr_out = xlmr_model(**xlmr_inputs)
        # Mean pooling
        attn_mask = xlmr_inputs["attention_mask"]
        token_embs = xlmr_out.last_hidden_state  # [B, seq, 768]
        mask_expanded = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
        sum_embs = (token_embs * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        xlmr_emb = (sum_embs / sum_mask).cpu()  # [B, 768]
        xlmr_embs.append(xlmr_emb)

        # ── Progress ──
        elapsed = time.time() - t0
        pct = batch_end / N * 100
        print(
            f"\r   🔄 Progress: {batch_end}/{N} ({pct:.0f}%) "
            f"| ⏱️ {elapsed:.0f}s "
            f"| 🖼️ imgs found: {n_img_found}, missing: {n_img_missing}",
            end="", flush=True,
        )

    print()  # newline after progress

    # ── Concatenate everything ─────────────────────────────────────────────
    result = {
        "ids": ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "splits": splits,
        "clip_text_emb": torch.cat(clip_text_embs, dim=0),      # [N, 512]
        "clip_image_emb": torch.cat(clip_image_embs, dim=0),    # [N, 512]
        "xlmr_emb": torch.cat(xlmr_embs, dim=0),                # [N, 768]
        "clip_score": torch.cat(clip_scores, dim=0),             # [N]
        "old_text_feats": torch.tensor(old_text_feats, dtype=torch.float32),  # [N, 3]
        "image_found": image_found_flags,
    }

    total_time = time.time() - t0
    print(f"\n✅ Feature extraction complete in {total_time:.1f}s")
    print(f"   CLIP text:  {result['clip_text_emb'].shape}")
    print(f"   CLIP image: {result['clip_image_emb'].shape}")
    print(f"   XLM-R text: {result['xlmr_emb'].shape}")
    print(f"   CLIP score: {result['clip_score'].shape}")
    print(f"   Old feats:  {result['old_text_feats'].shape}")
    print(f"   Labels dist: REAL={int((result['labels']==0).sum())}, FAKE={int((result['labels']==1).sum())}")
    print(f"   Images found: {n_img_found}/{N} ({n_img_found/N*100:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract CLIP+XLMR features")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "03_clean" / "Fakeddit" / "labeled_master_binary.jsonl"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "04_features" / "Fakeddit" / "features_v2.pt"))
    args = parser.parse_args()

    print("=" * 65)
    print("🚀 FEATURE EXTRACTOR v2 — CLIP + XLM-RoBERTa")
    print("=" * 65)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = output_path.parent

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Load data
    print(f"\n📂 Loading dataset: {input_path}")
    records = load_dataset(str(input_path))
    print(f"   ✅ Loaded {len(records)} records")

    # Extract features
    features = extract_all_features(records, batch_size=16)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, str(output_path))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 Saved features to: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")

    print("\n" + "=" * 65)
    print("✅ Extraction Complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()
