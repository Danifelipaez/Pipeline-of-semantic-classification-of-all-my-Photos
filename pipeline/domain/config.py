from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "source": "./photos",
    "output": "./sorted",
    "operation": "copy",
    "categories": [
        "nature",
        "birds",
        "street photography",
        "family",
        "portraits",
        "architecture",
        "food",
        "events",
    ],
    "ollama": {
        "url": "http://localhost:11434",
        "model": "gemma4",
        "timeout_seconds": 120,
    },
    "classification_prompt": (
        "You are an advanced image classification model specialized in semantic pattern recognition. "
        "Analyze the image carefully and choose exactly one category from this list: {categories}. "
        "Only respond with the category name. "
        "If you are not at least 85% confident, respond with 'uncategorized'. "
        "If the image contains multiple patterns, choose the most prominent one. "
        "Be strict and avoid guessing. "
        "Confidence: Output the category name, then a pipe '|', then the confidence as a percentage (e.g., 'nature|92')."
    ),
    "preprocessing": {
        "max_side": 1024,
        "jpeg_quality": 85,
    },
    "classification": {
        "min_confidence": 0.85,
        "fallback_category": "uncategorized",
        "require_confidence_format": True,
        "workers_mode": "auto",
        "num_workers": 4,
        "max_workers_cap": 6,
    },
}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate config.yaml preserving backward compatibility."""
    config = {
        **DEFAULT_CONFIG,
        "ollama": dict(DEFAULT_CONFIG["ollama"]),
        "preprocessing": dict(DEFAULT_CONFIG["preprocessing"]),
        "classification": dict(DEFAULT_CONFIG["classification"]),
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}

        for key, value in loaded.items():
            if key in {"ollama", "preprocessing", "classification"} and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    if not config.get("categories"):
        raise ValueError("Config must define at least one category")

    config["categories"] = [str(c).strip() for c in config["categories"] if str(c).strip()]
    if not config["categories"]:
        raise ValueError("Config categories are empty")

    operation = str(config.get("operation", "copy")).lower()
    if operation not in {"copy", "move"}:
        raise ValueError("operation must be 'copy' or 'move'")
    config["operation"] = operation

    min_confidence = float(config["classification"].get("min_confidence", 0.85))
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError("classification.min_confidence must be between 0 and 1")
    config["classification"]["min_confidence"] = min_confidence

    fallback_category = str(config["classification"].get("fallback_category", "uncategorized")).strip()
    if not fallback_category:
        raise ValueError("classification.fallback_category cannot be empty")
    config["classification"]["fallback_category"] = fallback_category
    config["classification"]["require_confidence_format"] = bool(
        config["classification"].get("require_confidence_format", True)
    )

    workers_mode = str(config["classification"].get("workers_mode", "auto")).strip().lower()
    if workers_mode not in {"auto", "manual"}:
        raise ValueError("classification.workers_mode must be 'auto' or 'manual'")
    config["classification"]["workers_mode"] = workers_mode

    num_workers = int(config["classification"].get("num_workers", 4))
    if num_workers < 1 or num_workers > 64:
        raise ValueError("classification.num_workers must be between 1 and 64")
    config["classification"]["num_workers"] = num_workers

    max_workers_cap = int(config["classification"].get("max_workers_cap", 6))
    if max_workers_cap < 1 or max_workers_cap > 64:
        raise ValueError("classification.max_workers_cap must be between 1 and 64")
    config["classification"]["max_workers_cap"] = max_workers_cap

    return config
