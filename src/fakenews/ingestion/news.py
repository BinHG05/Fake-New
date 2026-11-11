"""
News ingestion placeholder and utilities for external news articles.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_news_item(title: str, body: str, source: str = "news") -> dict[str, Any]:
    """Wraps raw news article into standardized pipeline format."""
    return {
        "title": title,
        "clean_text": body,
        "source": source,
        "cascade": [],
    }

