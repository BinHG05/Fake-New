"""
Auto-Labeling Tool for Fake News Detection
============================================
Hỗ trợ 2 phương pháp:
  1. text-only: Zero-shot text classification (BART-MNLI) — chỉ dùng text
  2. multimodal: CLIP (text + image) — dùng cả text lẫn ảnh ← khuyến nghị

Modes:
  - binary: True / Fake
  - fine:   TRUE / MOSTLY_TRUE / HALF_TRUE / BARELY_TRUE / FALSE / PANTS_ON_FIRE

Usage:
    # CLIP multimodal (recommended)
    python src/utils/auto_labeler.py --input data.jsonl --method clip --mode binary
    python src/utils/auto_labeler.py --input data.jsonl --method clip --mode fine --limit 10

    # Text-only fallback
    python src/utils/auto_labeler.py --input data.jsonl --method text --mode fine

    # Export Label Studio predictions
    python src/utils/auto_labeler.py --input data.jsonl --method clip --ls-predictions

    # Analyze results
    python src/utils/auto_labeler.py --analyze data_auto_labeled.jsonl

    # Evaluate accuracy on labeled data
    python src/utils/auto_labeler.py --evaluate data.jsonl --method clip --mode binary
"""

import json
import os
import sys
import argparse
import time
import base64
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# ============================================================
# Label & Prompt Definitions
# ============================================================

FINE_LABELS = ["TRUE", "MOSTLY_TRUE", "HALF_TRUE", "BARELY_TRUE", "FALSE", "PANTS_ON_FIRE"]
BINARY_LABELS = ["True", "Fake"]

# --- CLIP prompts: short, visual-friendly descriptions ---
CLIP_FINE_PROMPTS = {
    "TRUE":         [
        "a real authentic photograph with a truthful accurate caption",
        "genuine unedited news photo with factual description",
        "real photograph showing exactly what the title describes",
    ],
    "MOSTLY_TRUE":  [
        "a mostly real photo with a slightly exaggerated caption",
        "genuine photograph with a mostly accurate but slightly misleading title",
    ],
    "HALF_TRUE":    [
        "an image with a partially true but missing context caption",
        "a real photo used with a misleading or incomplete title",
    ],
    "BARELY_TRUE":  [
        "a photo with a mostly misleading caption that has a tiny grain of truth",
        "an image used out of context to support a false narrative",
    ],
    "FALSE":        [
        "a fake manipulated image with a false caption",
        "a digitally altered or fabricated photo with false information",
        "misleading fake news image designed to spread misinformation",
    ],
    "PANTS_ON_FIRE": [
        "an obviously fake satirical meme or absurd fabrication",
        "outrageous fake satirical image clearly made up as a joke or lie",
        "ridiculous photoshopped meme that is completely fabricated",
    ],
}

CLIP_BINARY_PROMPTS = {
    "True": [
        "a real authentic photograph with a truthful accurate caption",
        "genuine unedited news photo with factual description",
        "real photograph showing exactly what the title describes",
        "authentic real-world photo with honest title",
    ],
    "Fake": [
        "a fake manipulated image with a false misleading caption",
        "satirical meme or joke image not meant to be taken seriously",
        "digitally altered photoshopped fabricated image",
        "misleading out-of-context image designed to deceive",
    ],
}

# --- Zero-shot text hypotheses (for text-only fallback) ---
TEXT_FINE_HYPOTHESES = {
    "TRUE":         "This news article or social media post contains completely true and verified information.",
    "MOSTLY_TRUE":  "This post is mostly accurate but may contain minor inaccuracies.",
    "HALF_TRUE":    "This post is partially true but missing important context or details.",
    "BARELY_TRUE":  "This post contains a small element of truth but is mostly misleading.",
    "FALSE":        "This post contains false or fabricated information.",
    "PANTS_ON_FIRE": "This post is an outrageous lie, completely fabricated, or absurd satire.",
}

TEXT_BINARY_HYPOTHESES = {
    "True": "This news or social media post contains real, genuine, truthful information.",
    "Fake": "This post contains fake, misleading, fabricated, or satirical information.",
}


# ============================================================
# CLIP Multimodal Classifier
# ============================================================

class CLIPClassifier:
    """Multimodal classifier using CLIP — processes both text + image."""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="auto"):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from PIL import Image
        except ImportError:
            print("❌ Cần cài: pip install transformers torch Pillow")
            sys.exit(1)

        self.torch = torch
        self.Image = Image

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            print(f"🚀 Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Sử dụng CPU")

        print(f"📥 Đang tải CLIP model: {model_name}...")
        start = time.time()
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        elapsed = time.time() - start
        print(f"✅ CLIP model sẵn sàng ({elapsed:.1f}s)")

        # Pre-compute prompt embeddings for efficiency
        self._prompt_cache = {}

    def _get_text_embeddings(self, texts):
        """Encode a list of texts into CLIP embedding space."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        with self.torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            # Handle both tensor and BaseModelOutput return types
            if hasattr(emb, 'pooler_output'):
                emb = emb.pooler_output
            elif hasattr(emb, 'last_hidden_state'):
                emb = emb.last_hidden_state[:, 0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def _get_prompt_embeddings(self, mode):
        """Get cached prompt embeddings for label descriptions."""
        if mode in self._prompt_cache:
            return self._prompt_cache[mode]

        prompts = CLIP_BINARY_PROMPTS if mode == "binary" else CLIP_FINE_PROMPTS
        labels = BINARY_LABELS if mode == "binary" else FINE_LABELS

        # Average multiple prompts per label
        label_embeddings = []
        for label in labels:
            embs = self._get_text_embeddings(prompts[label])
            avg_emb = embs.mean(dim=0, keepdim=True)
            avg_emb = avg_emb / avg_emb.norm(dim=-1, keepdim=True)
            label_embeddings.append(avg_emb)

        result = self.torch.cat(label_embeddings, dim=0)  # [num_labels, dim]
        self._prompt_cache[mode] = (labels, result)
        return labels, result

    def classify(self, text, image_path, mode="binary"):
        """
        Classify using CLIP multimodal.

        Strategy:
        1. If image exists: compute image-to-prompt similarity (weight 0.7)
           + text-to-prompt similarity (weight 0.3)
        2. If no image: use text-to-prompt similarity only

        Returns: {"label": str, "confidence": float, "all_scores": dict,
                  "used_image": bool}
        """
        labels, prompt_embs = self._get_prompt_embeddings(mode)
        used_image = False
        scores_combined = None

        # --- Image embedding ---
        if image_path and os.path.exists(image_path):
            try:
                image = self.Image.open(image_path).convert("RGB")
                img_inputs = self.processor(images=image, return_tensors="pt")
                img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()
                              if k == "pixel_values"}
                with self.torch.no_grad():
                    img_emb = self.model.get_image_features(**img_inputs)
                    # Handle both tensor and BaseModelOutput return types
                    if hasattr(img_emb, 'pooler_output'):
                        img_emb = img_emb.pooler_output
                    elif hasattr(img_emb, 'last_hidden_state'):
                        img_emb = img_emb.last_hidden_state[:, 0]
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                # cosine similarity: image vs label prompts
                img_scores = (img_emb @ prompt_embs.T).squeeze(0)  # [num_labels]
                used_image = True
            except Exception as e:
                img_scores = None
        else:
            img_scores = None

        # --- Text embedding ---
        if text and text.strip():
            txt_emb = self._get_text_embeddings([text])
            txt_scores = (txt_emb @ prompt_embs.T).squeeze(0)  # [num_labels]
        else:
            txt_scores = None

        # --- Combine scores ---
        if img_scores is not None and txt_scores is not None:
            # Multimodal: weight image higher since text is short
            scores_combined = 0.7 * img_scores + 0.3 * txt_scores
        elif img_scores is not None:
            scores_combined = img_scores
        elif txt_scores is not None:
            scores_combined = txt_scores
        else:
            return {"label": "Unlabeled", "confidence": 0.0, "all_scores": {},
                    "used_image": False}

        # Softmax to normalize
        probs = self.torch.softmax(scores_combined * 100, dim=0)  # temperature scaling

        all_scores = {}
        for i, label in enumerate(labels):
            all_scores[label] = round(probs[i].item(), 4)

        best_idx = probs.argmax().item()
        best_label = labels[best_idx]
        best_confidence = probs[best_idx].item()

        return {
            "label": best_label,
            "confidence": round(best_confidence, 4),
            "all_scores": all_scores,
            "used_image": used_image,
        }


# ============================================================
# Zero-shot Text-Only Classifier (Fallback)
# ============================================================

class TextClassifier:
    """Text-only classifier using BART zero-shot NLI."""

    def __init__(self, model_name="facebook/bart-large-mnli", device="auto"):
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            print("❌ Cần cài: pip install transformers torch")
            sys.exit(1)

        if device == "auto":
            device_id = 0 if torch.cuda.is_available() else -1
            if device_id == 0:
                print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 CPU mode")
        elif device == "cpu":
            device_id = -1
        else:
            device_id = int(device)

        print(f"📥 Đang tải model: {model_name}...")
        start = time.time()
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device_id,
                                    model_kwargs={"use_safetensors": True})
        self.model_name = model_name
        print(f"✅ Model sẵn sàng ({time.time() - start:.1f}s)")

    def classify(self, text, image_path=None, mode="fine"):
        """Text-only classification. image_path is ignored."""
        if not text or not text.strip():
            return {"label": "Unlabeled", "confidence": 0.0, "all_scores": {},
                    "used_image": False}

        hypotheses_map = TEXT_BINARY_HYPOTHESES if mode == "binary" else TEXT_FINE_HYPOTHESES
        labels = list(hypotheses_map.keys())
        hypotheses = list(hypotheses_map.values())

        result = self.classifier(text, candidate_labels=hypotheses, multi_label=False)

        scores = {}
        for hyp, score in zip(result["labels"], result["scores"]):
            for label_name, label_hyp in hypotheses_map.items():
                if hyp == label_hyp:
                    scores[label_name] = round(score, 4)
                    break

        best_label = max(scores, key=scores.get)
        return {
            "label": best_label,
            "confidence": scores[best_label],
            "all_scores": scores,
            "used_image": False,
        }


# ============================================================
# Groq LLM Vision Classifier (High Accuracy)
# ============================================================

class GroqClassifier:
    """High-accuracy classifier using Groq API with Llama Vision model."""

    SYSTEM_PROMPT_BINARY = """You are an expert fact-checker and misinformation analyst.
You will be given a social media post with an IMAGE and a TEXT TITLE.
Your task: classify the post as True or Fake.

- True: The image and text represent genuine, real information. The image is unedited and the caption accurately describes it.
- Fake: The image or text is misleading, manipulated, satirical, out-of-context, or fabricated.

Analyze BOTH the image AND the text carefully. Consider:
1. Does the image look authentic or manipulated/photoshopped?
2. Is the title consistent with what the image shows?
3. Does this look like satire, a meme, or a joke?
4. Is the image used out of context?

Respond ONLY with valid JSON:
{"label": "True" or "Fake", "confidence": 0.0-1.0, "reason": "brief explanation"}"""

    SYSTEM_PROMPT_FINE = """You are an expert fact-checker and misinformation analyst.
You will be given a social media post with an IMAGE and a TEXT TITLE (caption).
Your task: classify the truthfulness into one of 6 categories.

Categories (from most true to most false):
- TRUE: Completely true, verified information. Authentic image with accurate caption.
- MOSTLY_TRUE: Mostly accurate but may have minor inaccuracies or slight exaggeration.
- HALF_TRUE: Partially true but missing important context or details.
- BARELY_TRUE: Contains a small element of truth but is mostly misleading or out of context.
- FALSE: Contains false, fabricated, or significantly misleading information. Also applies if the user caption is denying a factual event through sarcasm.
- PANTS_ON_FIRE: Outrageous lie, absurd satire, obvious meme/joke, or unhinged conspiracy theory.

CRITICAL INSTRUCTIONS FOR SARCASM AND CONSPIRACY:
- The text caption may be sarcastic or denying the event shown in the image (e.g., "they think you'll believe this").
- Even if the news article in the image is FACTUALLY TRUE, if the user's caption is MOCKING it, DENYING it, or claiming it's a "false flag" / conspiracy, you MUST label the post as FALSE or PANTS_ON_FIRE. The truthfulness is based on the USER'S INTENT in the caption, not just the image content alone.

Analyze BOTH the image AND the text carefully. Consider:
1. Is the user's caption asserting a truth, or is it doubting/denying the factual image?
2. Does the image look authentic or manipulated?
3. Does this look like a conspiracy theory, a meme, or a joke?

Respond ONLY with valid JSON:
{"label": "ONE_OF_THE_6_LABELS", "confidence": 0.0-1.0, "reason": "brief explanation"}"""

    def __init__(self, model_name=None, device="auto"):
        """Initialize Groq API client."""
        # Load API key
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            env_file = os.path.join(project_root, ".env")
            if os.path.exists(env_file):
                with open(env_file, "r") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            self.api_key = line.strip().split("=", 1)[1]
                            break

        if not self.api_key:
            print("❌ GROQ_API_KEY not found! Set in .env or environment variable.")
            sys.exit(1)

        self.model_name = model_name or "meta-llama/llama-4-scout-17b-16e-instruct"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"✅ Groq API ready | Model: {self.model_name}")

    def _encode_image(self, image_path):
        """Encode image to base64 data URL."""
        ext = os.path.splitext(image_path)[1].lower()
        mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        mime_type = mime.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def classify(self, text, image_path=None, mode="binary"):
        """Classify using Groq LLM Vision API."""
        import urllib.request
        import urllib.error

        system_prompt = self.SYSTEM_PROMPT_BINARY if mode == "binary" else self.SYSTEM_PROMPT_FINE

        # Build message content
        content = []
        used_image = False

        if image_path and os.path.exists(image_path):
            try:
                data_url = self._encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
                used_image = True
            except Exception as e:
                print(f"  ⚠️ Image load failed: {e}")

        user_text = f"Title: {text}" if text else "(no title)"
        content.append({"type": "text", "text": user_text})

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            reply = result["choices"][0]["message"]["content"].strip()

            # Parse JSON from response
            parsed = self._parse_response(reply, mode)
            parsed["used_image"] = used_image
            return parsed

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace') if hasattr(e, 'read') else ''
            print(f"  ❌ Groq API error {e.code}: {error_body[:200]}")
            return {"label": "Unlabeled", "confidence": 0.0,
                    "all_scores": {}, "used_image": False}
        except Exception as e:
            print(f"  ❌ Groq API error: {e}")
            return {"label": "Unlabeled", "confidence": 0.0,
                    "all_scores": {}, "used_image": False}

    def _parse_response(self, reply, mode):
        """Parse LLM JSON response."""
        valid_labels = BINARY_LABELS if mode == "binary" else FINE_LABELS

        try:
            # Find JSON in response
            start = reply.find("{")
            end = reply.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(reply[start:end])
                label = data.get("label", "Unlabeled")
                confidence = float(data.get("confidence", 0.5))

                # Map label
                if label not in valid_labels:
                    label_upper = label.upper().replace(" ", "_")
                    for vl in valid_labels:
                        if vl.upper() == label_upper:
                            label = vl
                            break

                if label in valid_labels:
                    scores = {l: (confidence if l == label else
                               round((1 - confidence) / (len(valid_labels) - 1), 4))
                              for l in valid_labels}
                    return {"label": label, "confidence": round(confidence, 4),
                            "all_scores": scores}

        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: search text for label
        for label in valid_labels:
            if label.lower() in reply.lower():
                return {"label": label, "confidence": 0.5,
                        "all_scores": {l: (0.5 if l == label else
                                     round(0.5 / (len(valid_labels) - 1), 4))
                                     for l in valid_labels}}

        return {"label": "Unlabeled", "confidence": 0.0, "all_scores": {}}


# ============================================================
# Main Processing
# ============================================================

def resolve_image_path(record, project_root_dir):
    """Resolve image path from record's image_info."""
    image_info = record.get("image_info", {})
    processed_path = image_info.get("processed_path", "")
    if not processed_path:
        return None

    # Normalize path separators
    processed_path = processed_path.replace("\\", "/")

    # Try relative to project root
    full_path = os.path.join(project_root_dir, processed_path)
    if os.path.exists(full_path):
        return full_path

    # Try as absolute path
    if os.path.exists(processed_path):
        return processed_path

    return None


def process_jsonl(
    input_path,
    output_path=None,
    mode="binary",
    method="clip",
    threshold=0.3,
    model_name=None,
    device="auto",
    limit=None,
    only_unlabeled=True,
    ls_predictions=False,
):
    """Process a JSONL file and auto-label each record."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            total = kwargs.get("total", "?")
            for i, item in enumerate(iterable):
                if i % 10 == 0:
                    print(f"  Processing {i}/{total}...", end="\r")
                yield item
            print()

    input_file = Path(input_path)
    if output_path is None:
        if input_file.name.endswith("_auto_labeled.jsonl"):
            # Trực tiếp là file đầu ra
            output_file = input_file
        else:
            stem = input_file.stem
            output_file = input_file.parent / f"{stem}_auto_labeled.jsonl"
    else:
        output_file = Path(output_path)

    # Đọc từ output_file trước nếu đã có (để tiếp tục nối file), nếu không thì đọc từ input
    read_file = output_file if output_file.exists() else input_file
    if not read_file.exists():
        print(f"❌ File không tồn tại: {read_file}")
        return

    print(f"📥 Đang đọc dữ liệu từ: {read_file.name}")
    # Read records
    records = []
    with open(read_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total = len(records)
    print(f"📊 Tổng số {total} dòng trong dữ liệu")

    # Stats
    unlabeled_count = sum(
        1 for r in records if r.get("label", "") in ["Unlabeled", "", None]
    )
    has_image = sum(
        1 for r in records if r.get("image_info", {}).get("processed_path")
    )
    print(f"🏷️  Unlabeled: {unlabeled_count}/{len(records)}")
    print(f"🖼️  Có ảnh: {has_image}/{len(records)}")
    print(f"🔧 Chế độ: {'Binary (True/Fake)' if mode == 'binary' else '6-class fine'}")
    method_names = {"clip": "CLIP multimodal", "text": "Text-only zero-shot", "groq": "Groq LLM Vision"}
    print(f"🧠 Phương pháp: {method_names.get(method, method)}")
    print(f"📏 Threshold: {threshold}")

    # Load classifier
    if method == "groq":
        classifier = GroqClassifier(model_name, device)
    elif method == "clip":
        if model_name is None:
            model_name = "openai/clip-vit-base-patch32"
        classifier = CLIPClassifier(model_name, device)
    else:
        if model_name is None:
            model_name = "facebook/bart-large-mnli"
        classifier = TextClassifier(model_name, device)

    # Process
    labeled_count = 0
    skipped_count = 0
    below_threshold = 0
    image_used_count = 0
    ls_prediction_data = []

    # Lọc ra các record cần xử lý (chưa có nhãn)
    records_to_process = []
    for record in records:
        current_label = record.get("label", "Unlabeled")
        if only_unlabeled and current_label not in ["Unlabeled", "", None]:
            skipped_count += 1
            continue
        records_to_process.append(record)

    # Giới hạn số lượng nếu có tham số limit
    if limit:
        records_to_process = records_to_process[:limit]
        print(f"🎯 Sẽ xử lý {len(records_to_process)} dòng chưa có nhãn (limit={limit})")

    for record in tqdm(records_to_process, desc="🤖 Auto-labeling", total=len(records_to_process)):

        text = record.get("clean_text", "") or record.get("raw_text", "")
        image_path = resolve_image_path(record, project_root)

        # Classify
        result = classifier.classify(text, image_path, mode)

        if result.get("used_image"):
            image_used_count += 1

        if result["confidence"] >= threshold:
            record["label"] = result["label"]
            record["auto_label"] = True
            record["auto_label_confidence"] = result["confidence"]
            record["auto_label_scores"] = result["all_scores"]
            record["auto_label_method"] = method
            record["auto_label_model"] = classifier.model_name
            record["auto_label_used_image"] = result.get("used_image", False)
            record["auto_label_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            labeled_count += 1
        else:
            record["auto_label"] = True
            record["auto_label_confidence"] = result["confidence"]
            record["auto_label_scores"] = result["all_scores"]
            record["auto_label_method"] = method
            record["auto_label_model"] = classifier.model_name
            record["auto_label_used_image"] = result.get("used_image", False)
            record["auto_label_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            below_threshold += 1

        # Label Studio predictions
        if ls_predictions:
            ls_pred = build_ls_prediction(record, result, mode)
            if ls_pred:
                ls_prediction_data.append(ls_pred)

    if output_path is not None:
        final_output_path = str(output_path)
    else:
        final_output_path = str(output_file)

    # Write JSONL
    with open(final_output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*55}")
    print(f"📊 KẾT QUẢ AUTO-LABELING ({method.upper()})")
    print(f"{'='*55}")
    print(f"  ✅ Đã gán nhãn:          {labeled_count}")
    print(f"  ⚠️  Dưới threshold:       {below_threshold}")
    print(f"  ⏭️  Bỏ qua (đã có nhãn):  {skipped_count}")
    if method == "clip":
        print(f"  🖼️  Sử dụng ảnh:         {image_used_count}/{labeled_count + below_threshold}")
    print(f"  📁 Output: {output_path}")

    # Write LS predictions
    if ls_predictions and ls_prediction_data:
        ls_output = str(Path(final_output_path).with_suffix("")) + "_ls_predictions.json"
        
        # Merge với file LS cũ nếu có
        existing_ls_data = []
        if os.path.exists(ls_output):
            try:
                with open(ls_output, "r", encoding="utf-8") as f:
                    existing_ls_data = json.load(f)
            except Exception:
                pass
                
        # Dùng dictionary để merge theo ID, tránh trùng lặp
        ls_dict = {str(item.get("data", {}).get("id", i)): item for i, item in enumerate(existing_ls_data)}
        for item in ls_prediction_data:
            ls_dict[str(item.get("data", {}).get("id", ""))] = item
            
        final_ls_data = list(ls_dict.values())

        with open(ls_output, "w", encoding="utf-8") as f:
            json.dump(final_ls_data, f, ensure_ascii=False, indent=2)
        print(f"  📁 LS Predictions: {ls_output} (Tổng: {len(final_ls_data)} bài)")

    print(f"{'='*55}")
    return final_output_path


def build_ls_prediction(record, result, mode):
    """Build Label Studio prediction entry."""
    if result["confidence"] <= 0:
        return None

    data = {}
    for key in ["id", "timestamp", "label", "original_label", "clean_text",
                 "raw_text", "text_features", "image_info", "user_features",
                 "graph_features", "source_dataset", "processed_timestamp",
                 "user_id", "split"]:
        if key in record:
            data[key] = record[key]

    # Image path for Label Studio
    image_info = record.get("image_info", {})
    processed_path = image_info.get("processed_path", "")
    if processed_path and not processed_path.startswith("/data/local-files"):
        clean_path = processed_path.replace("\\", "/")
        if clean_path.startswith("data/"):
            clean_path = clean_path[5:]
        data["image"] = f"/data/local-files/?d=data/{clean_path}"

    return {
        "data": data,
        "predictions": [{
            "model_version": f"auto-labeler-{result.get('used_image', False) and 'clip' or 'text'}-v1",
            "result": [{
                "from_name": "label_fine",
                "to_name": "post_text",
                "type": "choices",
                "value": {"choices": [result["label"]]},
            }],
            "score": result["confidence"],
        }],
    }


# ============================================================
# Evaluation (accuracy on labeled data)
# ============================================================

def evaluate_accuracy(input_path, method="clip", mode="binary",
                      model_name=None, device="auto", limit=None):
    """
    Test auto-labeler accuracy against ground-truth labels.
    Only tests records that already have True/Fake or fine labels.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            for item in iterable:
                yield item

    input_file = Path(input_path)
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Only keep records with valid labels
    valid_labels = set(BINARY_LABELS if mode == "binary" else FINE_LABELS)
    labeled_records = [r for r in records if r.get("original_label", r.get("label")) in valid_labels]

    if not labeled_records:
        print("❌ Không tìm thấy records có nhãn hợp lệ để test!")
        return

    if limit:
        labeled_records = labeled_records[:limit]

    print(f"📊 Test {len(labeled_records)} records có nhãn | Mode: {mode} | Method: {method}")

    # Load classifier
    if method == "clip":
        classifier = CLIPClassifier(model_name or "openai/clip-vit-base-patch32", device)
    else:
        classifier = TextClassifier(model_name or "facebook/bart-large-mnli", device)

    correct = 0
    total = 0
    confusion = {}  # (true_label, pred_label) → count
    image_used = 0

    for record in tqdm(labeled_records, desc="🧪 Evaluating"):
        true_label = record.get("original_label", record.get("label"))
        text = record.get("clean_text", "") or record.get("raw_text", "")
        image_path = resolve_image_path(record, project_root) if method == "clip" else None

        result = classifier.classify(text, image_path, mode)
        pred_label = result["label"]

        if result.get("used_image"):
            image_used += 1

        total += 1
        if pred_label == true_label:
            correct += 1

        key = (true_label, pred_label)
        confusion[key] = confusion.get(key, 0) + 1

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*55}")
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ ({method.upper()})")
    print(f"{'='*55}")
    print(f"  Tổng test:    {total}")
    print(f"  Đúng:         {correct}")
    print(f"  Accuracy:     {accuracy:.1%}")
    if method == "clip":
        print(f"  Dùng ảnh:     {image_used}/{total}")

    # Confusion matrix
    labels = sorted(valid_labels)
    print(f"\n  📋 Confusion Matrix:")
    print(f"  {'':15s}", end="")
    for l in labels:
        print(f" {l:>12s}", end="")
    print("  (predicted)")
    for true_l in labels:
        print(f"  {true_l:15s}", end="")
        for pred_l in labels:
            count = confusion.get((true_l, pred_l), 0)
            print(f" {count:>12d}", end="")
        print()
    print(f"  (actual)")

    # Per-class accuracy
    print(f"\n  📊 Per-class:")
    for label in labels:
        total_class = sum(confusion.get((label, p), 0) for p in labels)
        correct_class = confusion.get((label, label), 0)
        acc_class = correct_class / total_class if total_class > 0 else 0
        print(f"    {label:15s}  {correct_class}/{total_class}  ({acc_class:.1%})")

    print(f"{'='*55}")
    return accuracy


# ============================================================
# Analysis
# ============================================================

def analyze_results(input_path):
    """Analyze auto-labeling results."""
    from collections import Counter

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    auto_labeled = [r for r in records if r.get("auto_label")]
    labeled = [r for r in auto_labeled if r.get("label") not in ["Unlabeled", "", None]]
    still_unlabeled = [r for r in records if r.get("label") in ["Unlabeled", "", None]]

    print(f"\n📊 PHÂN TÍCH KẾT QUẢ AUTO-LABELING")
    print(f"{'='*50}")
    print(f"  Tổng records:       {len(records)}")
    print(f"  Auto-labeled:       {len(labeled)}")
    print(f"  Còn Unlabeled:      {len(still_unlabeled)}")

    if labeled:
        label_dist = Counter(r["label"] for r in labeled)
        print(f"\n  📊 Phân bố nhãn:")
        for label, count in label_dist.most_common():
            pct = count / len(labeled) * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:15s} {count:4d} ({pct:5.1f}%) {bar}")

        # Confidence
        confidences = [r.get("auto_label_confidence", 0) for r in labeled]
        print(f"\n  📈 Confidence: avg={sum(confidences)/len(confidences):.3f}, "
              f"min={min(confidences):.3f}, max={max(confidences):.3f}")

        # Method stats
        methods = Counter(r.get("auto_label_method", "?") for r in labeled)
        print(f"  🧠 Method: {dict(methods)}")

        img_used = sum(1 for r in labeled if r.get("auto_label_used_image"))
        print(f"  🖼️  Image used: {img_used}/{len(labeled)}")

    print(f"{'='*50}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="🤖 Auto-Labeling Tool — Fake News Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLIP multimodal (recommended)
  python src/utils/auto_labeler.py --input data.jsonl --method clip --mode binary
  python src/utils/auto_labeler.py --input data.jsonl --method clip --mode fine --limit 10

  # Text-only fallback
  python src/utils/auto_labeler.py --input data.jsonl --method text --mode binary

  # Export Label Studio predictions
  python src/utils/auto_labeler.py --input data.jsonl --method clip --ls-predictions

  # Evaluate accuracy
  python src/utils/auto_labeler.py --evaluate data.jsonl --method clip --mode binary

  # Analyze results
  python src/utils/auto_labeler.py --analyze data_auto_labeled.jsonl
        """,
    )

    parser.add_argument("--input", help="Path to input JSONL file")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--method", choices=["clip", "text", "groq"], default="clip",
                        help="Classification method: groq (LLM vision, best), clip (multimodal), text (zero-shot NLI). Default: clip")
    parser.add_argument("--mode", choices=["fine", "binary"], default="binary",
                        help="Label mode: fine (6-class) or binary (True/Fake). Default: binary")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Min confidence threshold. Default: 0.3")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name. Default: auto per method")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, or cuda. Default: auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max records to process")
    parser.add_argument("--all", action="store_true", dest="process_all",
                        help="Process ALL records, not just Unlabeled")
    parser.add_argument("--ls-predictions", action="store_true",
                        help="Export Label Studio predictions JSON")
    parser.add_argument("--analyze", metavar="FILE",
                        help="Analyze results from auto-labeled JSONL")
    parser.add_argument("--evaluate", metavar="FILE",
                        help="Evaluate accuracy on labeled data")

    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.analyze)
        return

    if args.evaluate:
        evaluate_accuracy(
            args.evaluate,
            method=args.method,
            mode=args.mode,
            model_name=args.model,
            device=args.device,
            limit=args.limit,
        )
        return

    if not args.input:
        parser.print_help()
        print("\n❌ Cần --input, --analyze, hoặc --evaluate")
        sys.exit(1)

    print("=" * 55)
    print("  🤖 AUTO-LABELING TOOL")
    print(f"  Method: {args.method.upper()} | Mode: {args.mode}")
    print("=" * 55)

    process_jsonl(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        method=args.method,
        threshold=args.threshold,
        model_name=args.model,
        device=args.device,
        limit=args.limit,
        only_unlabeled=not args.process_all,
        ls_predictions=args.ls_predictions,
    )


if __name__ == "__main__":
    main()
