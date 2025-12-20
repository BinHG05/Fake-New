"""
Embedding Extractor Module for Multimodal Fake News Detection.

Extracts embeddings from text (XLM-R/BERT) and images (CLIP).
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextEmbeddingExtractor:
    """
    Extract text embeddings using XLM-RoBERTa or multilingual BERT.
    
    Default: xlm-roberta-base (768 dim)
    """
    
    def __init__(
        self, 
        model_name: str = 'xlm-roberta-base',
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
            max_length: Maximum token length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return
        
        from transformers import AutoModel, AutoTokenizer
        
        print(f"📥 Loading text model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"✓ Text model loaded on {self.device}")
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 768  # XLM-R base
    
    @torch.no_grad()
    def extract(self, text: str) -> torch.Tensor:
        """
        Extract embedding for single text.
        
        Args:
            text: Input text string
            
        Returns:
            Tensor of shape [embedding_dim]
        """
        self._load_model()
        
        if not text or not isinstance(text, str):
            return torch.zeros(self.embedding_dim)
        
        inputs = self._tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        outputs = self._model(**inputs)
        
        # Mean pooling over sequence
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        embedding = sum_embeddings / sum_mask
        
        return embedding.squeeze(0).cpu()
    
    @torch.no_grad()
    def batch_extract(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Extract embeddings for batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape [N, embedding_dim]
        """
        self._load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Handle empty/invalid texts
            batch_texts = [t if t and isinstance(t, str) else "" for t in batch_texts]
            
            inputs = self._tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self._model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            
            batch_embeddings = sum_embeddings / sum_mask
            all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)


class ImageEmbeddingExtractor:
    """
    Extract image embeddings using CLIP.
    
    Default: openai/clip-vit-base-patch32 (512 dim)
    """
    
    def __init__(
        self,
        model_name: str = 'openai/clip-vit-base-patch32',
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace CLIP model name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is not None:
            return
        
        from transformers import CLIPModel, CLIPProcessor
        
        print(f"📥 Loading image model: {self.model_name}")
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"✓ Image model loaded on {self.device}")
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 512  # CLIP ViT-base
    
    @torch.no_grad()
    def extract(self, image_path: str) -> torch.Tensor:
        """
        Extract embedding for single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tensor of shape [embedding_dim]
        """
        self._load_model()
        
        from PIL import Image
        
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image not found: {image_path}")
            return torch.zeros(self.embedding_dim)
        
        try:
            image = Image.open(path).convert('RGB')
            inputs = self._processor(images=image, return_tensors='pt').to(self.device)
            
            outputs = self._model.get_image_features(**inputs)
            embedding = F.normalize(outputs, p=2, dim=-1)
            
            return embedding.squeeze(0).cpu()
        
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return torch.zeros(self.embedding_dim)
    
    @torch.no_grad()
    def batch_extract(self, image_paths: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        Extract embeddings for batch of images.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape [N, embedding_dim]
        """
        self._load_model()
        
        from PIL import Image
        
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            valid_indices = []
            
            for idx, path in enumerate(batch_paths):
                p = Path(path)
                if p.exists():
                    try:
                        img = Image.open(p).convert('RGB')
                        images.append(img)
                        valid_indices.append(idx)
                    except Exception as e:
                        logger.warning(f"Error loading {path}: {e}")
            
            # Initialize batch embeddings with zeros
            batch_embeddings = torch.zeros(len(batch_paths), self.embedding_dim)
            
            if images:
                inputs = self._processor(images=images, return_tensors='pt').to(self.device)
                outputs = self._model.get_image_features(**inputs)
                normalized = F.normalize(outputs, p=2, dim=-1).cpu()
                
                for new_idx, orig_idx in enumerate(valid_indices):
                    batch_embeddings[orig_idx] = normalized[new_idx]
            
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)


# Convenience function
def create_extractors(
    text_model: str = 'xlm-roberta-base',
    image_model: str = 'openai/clip-vit-base-patch32',
    device: Optional[str] = None
) -> tuple:
    """
    Create text and image extractors.
    
    Returns:
        (TextEmbeddingExtractor, ImageEmbeddingExtractor)
    """
    text_ext = TextEmbeddingExtractor(model_name=text_model, device=device)
    image_ext = ImageEmbeddingExtractor(model_name=image_model, device=device)
    return text_ext, image_ext