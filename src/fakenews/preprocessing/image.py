"""
Image preprocessing utilities for fake news multimodal detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def preprocess_image(
    image_path: str | Path,
    target_size: tuple[int, int] = (224, 224),
    output_path: str | Path | None = None,
) -> Optional[Image.Image]:
    """
    Loads, sanitizes, and prepares an image for CLIP/ViT processing:
    - Fixes EXIF rotation
    - Converts to RGB
    - Resizes to target dimension
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image path not found: %s", path)
        return None

    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = img.resize(target_size, Image.Resampling.BICUBIC)

            if output_path is not None:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                img.save(out, format="JPEG", quality=90)

            return img
    except Exception as e:
        logger.warning("Image preprocessing failed for %s: %s", path, e)
        return None

