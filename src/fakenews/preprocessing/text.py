"""
Text preprocessing utilities for fake news detection.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Cleans raw Reddit post / comment text:
    - Normalizes unicode (NFC)
    - Strips URLs
    - Cleans excessive whitespace and special characters
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFC", str(text))

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove markdown links [title](url) -> title
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

