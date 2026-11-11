"""
Configuration loader for FakeNews Detection Pipeline.
Loads YAML configurations and provides validated, dictionary/object access.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml


def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    If path is relative, resolves relative to project root.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "canonical.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = (root / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config

