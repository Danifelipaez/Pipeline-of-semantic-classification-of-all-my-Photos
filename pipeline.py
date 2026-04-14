from __future__ import annotations

import base64
import logging
import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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
        "model": "gemma3",
        "timeout_seconds": 120,
    },
    "classification_prompt": (
        "You are a photo classification model. "
        "Choose exactly one category from this list: {categories}. "
        "Respond with only the category name."
    ),
    "preprocessing": {
        "max_side": 1024,
        "jpeg_quality": 85,
    },
}


@dataclass
class Summary:
    counts: dict[str, int]
    total_images: int
    total_seconds: float
    avg_original_size: float
    avg_payload_size: float


def load_config(config_path: Path) -> dict[str, Any]:
    config = {
        **DEFAULT_CONFIG,
        "ollama": dict(DEFAULT_CONFIG["ollama"]),
        "preprocessing": dict(DEFAULT_CONFIG["preprocessing"]),
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for key, value in loaded.items():
            if key in {"ollama", "preprocessing"} and isinstance(value, dict):
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

    return config


def list_images(source: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def preprocess_image_for_inference(
    image_path: Path,
    *,
    max_side: int,
    jpeg_quality: int,
) -> tuple[str, int, int]:
    original_size = image_path.stat().st_size

    with Image.open(image_path) as img:
        if img.mode in {"RGBA", "P"}:
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        longest_side = max(width, height)
        if longest_side > max_side:
            scale = max_side / float(longest_side)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality)
        payload_bytes = buffer.getvalue()

    payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
    return payload_b64, original_size, len(payload_bytes)


def extract_category(response_text: str, categories: list[str]) -> str:
    normalized_response = response_text.lower().strip()
    normalized_response = re.sub(r"[^a-z0-9\s]", " ", normalized_response)
    normalized_response = re.sub(r"\s+", " ", normalized_response).strip()

    normalized_categories = {re.sub(r"\s+", " ", c.lower().strip()): c for c in categories}

    if normalized_response in normalized_categories:
        return normalized_categories[normalized_response]

    for normalized, original in normalized_categories.items():
        pattern = rf"\b{re.escape(normalized)}\b"
        if re.search(pattern, normalized_response):
            return original

    first_word = normalized_response.split(" ")[0] if normalized_response else ""
    for normalized, original in normalized_categories.items():
        if normalized.split(" ")[0] == first_word and first_word:
            return original

    return "uncategorized"


def classify_with_ollama(
    payload_b64: str,
    categories: list[str],
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    prompt = prompt_template.format(categories=", ".join(categories))

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [payload_b64],
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    output = response.json().get("response", "")
    return extract_category(str(output), categories)


def _configure_error_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"pipeline_errors_{log_file}")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def process_images(
    *,
    source: Path,
    output: Path,
    categories: list[str],
    operation: str,
    max_side: int,
    jpeg_quality: int,
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    dry_run: bool,
) -> Summary:
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    valid_categories = list(dict.fromkeys(categories + ["uncategorized"]))
    for category in valid_categories:
        (output / category).mkdir(parents=True, exist_ok=True)

    errors_logger = _configure_error_logger(output / "errors.log")

    images = list_images(source)
    counts: dict[str, int] = {category: 0 for category in valid_categories}
    total_original_size = 0
    total_payload_size = 0

    started = time.perf_counter()

    for image_path in images:
        category = "uncategorized"
        try:
            payload_b64, original_size, payload_size = preprocess_image_for_inference(
                image_path,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
            )
            try:
                category = classify_with_ollama(
                    payload_b64,
                    categories,
                    prompt_template,
                    ollama_url,
                    model,
                    timeout_seconds,
                )
            except Exception as exc:
                errors_logger.error("Failed to classify %s: %s", image_path, exc)
        except Exception as exc:
            errors_logger.error("Failed to preprocess %s: %s", image_path, exc)
            original_size = image_path.stat().st_size
            payload_size = 0

        total_original_size += original_size
        total_payload_size += payload_size

        if category not in valid_categories:
            category = "uncategorized"

        destination = output / category / image_path.name
        print(
            f"{image_path.name} | original_size={original_size} bytes "
            f"| payload_size={payload_size} bytes | category={category} "
            f"| destination={destination}"
        )

        counts[category] += 1
        if dry_run:
            continue

        if operation == "move":
            shutil.move(str(image_path), str(destination))
        else:
            shutil.copy2(image_path, destination)

    total_seconds = time.perf_counter() - started
    total_images = len(images)

    avg_original_size = (total_original_size / total_images) if total_images else 0.0
    avg_payload_size = (total_payload_size / total_images) if total_images else 0.0

    return Summary(
        counts=counts,
        total_images=total_images,
        total_seconds=total_seconds,
        avg_original_size=avg_original_size,
        avg_payload_size=avg_payload_size,
    )
