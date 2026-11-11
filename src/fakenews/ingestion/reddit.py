"""
Reddit crawler and comment tree fetcher.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
import requests

logger = logging.getLogger(__name__)


class RedditCrawler:
    """Fetches Reddit posts and comment cascades using standard HTTP endpoints."""

    def __init__(self, user_agent: str = "fakenews_portfolio_pipeline/1.0", timeout: int = 15) -> None:
        self.headers = {"User-Agent": user_agent}
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def fetch_post_and_comments(self, permalink_or_url: str) -> dict[str, Any]:
        """
        Fetches post details and comment cascade tree from Reddit permalink/URL.
        """
        url = permalink_or_url.strip()
        if not url.startswith("http"):
            url = f"https://www.reddit.com{url}"
        if not url.endswith(".json"):
            url = f"{url.rstrip('/')}.json"

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) < 2:
            raise ValueError(f"Unexpected Reddit JSON format for {url}")

        post_data = data[0]["data"]["children"][0]["data"]
        comments_raw = data[1]["data"]["children"]

        cascade: list[dict[str, Any]] = []

        def parse_comments(children: list[dict[str, Any]], parent_id: str) -> None:
            for child in children:
                if child.get("kind") != "t1":
                    continue
                c_data = child.get("data", {})
                c_id = c_data.get("id")
                body = c_data.get("body")
                if not c_id or not body:
                    continue
                cascade.append({
                    "id": c_id,
                    "parent_id": parent_id,
                    "text": body,
                    "author": c_data.get("author", "[deleted]"),
                    "score": c_data.get("score", 0),
                    "created_utc": c_data.get("created_utc", 0),
                })
                replies = c_data.get("replies")
                if isinstance(replies, dict):
                    sub_children = replies.get("data", {}).get("children", [])
                    parse_comments(sub_children, parent_id=c_id)

        parse_comments(comments_raw, parent_id=post_data.get("id", "root"))

        return {
            "id": post_data.get("id"),
            "title": post_data.get("title", ""),
            "author": post_data.get("author", "[deleted]"),
            "subreddit": post_data.get("subreddit", ""),
            "url": post_data.get("url", ""),
            "num_comments": post_data.get("num_comments", 0),
            "created_utc": post_data.get("created_utc", 0),
            "cascade": cascade,
        }

