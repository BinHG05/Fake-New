"""
Embedding extractors for Multimodal Fake News Detection.
Combines text embeddings (XLM-RoBERTa) and vision-language features (CLIP).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor:
    """
    Extracts 1025-dimensional multimodal features:
    [CLIP Text (512) | CLIP Image (512) | CLIP Cosine Similarity (1)]
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

        logger.info("Loading CLIP model: %s on %s...", self.model_name, self.device)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)

    @torch.no_grad()
    def extract(self, text: str, image_path: str | Path | None = None) -> torch.Tensor:
        self._load_model()
        from PIL import Image

        # 1. CLIP Text Embedding
        text_inputs = self._tokenizer(
            text or "",
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_outputs = self._model.text_model(**text_inputs)
        clip_text_out = self._model.text_projection(text_outputs.pooler_output)
        clip_text_out = F.normalize(clip_text_out, p=2, dim=-1)

        # 2. CLIP Image Embedding
        clip_image_out = torch.zeros(1, 512, device=self.device)
        if image_path is not None and os.path.exists(str(image_path)):
            try:
                img = Image.open(str(image_path)).convert("RGB")
                pixel_inputs = self._processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
                vision_outputs = self._model.vision_model(pixel_values=pixel_inputs)
                clip_image_out = self._model.visual_projection(vision_outputs.pooler_output)
                clip_image_out = F.normalize(clip_image_out, p=2, dim=-1)
            except Exception as e:
                logger.warning("CLIP Image extraction failed for %s: %s", image_path, e)

        # 3. CLIP Similarity Score
        score = (clip_text_out * clip_image_out).sum(dim=-1, keepdim=True)

        return torch.cat([clip_text_out, clip_image_out, score], dim=-1).squeeze(0).cpu()


class TextEmbeddingExtractor:
    """
    Extracts 768-dimensional text embeddings using XLM-RoBERTa.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading text model: %s on %s...", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def extract(self, text: str) -> torch.Tensor:
        self._load_model()
        inputs = self._tokenizer(
            text or "",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self._model(**inputs)
        # Mean pooling over attention mask
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled.squeeze(0).cpu()

