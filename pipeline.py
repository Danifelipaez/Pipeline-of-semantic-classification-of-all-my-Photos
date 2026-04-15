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


class OllamaError(Exception):
    """Raised when Ollama API is unreachable or fails to respond."""

    pass

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
    },
}


@dataclass
class Summary:
    counts: dict[str, int]
    total_images: int
    skipped_images: int
    total_seconds: float
    avg_original_size: float
    avg_payload_size: float


def load_config(config_path: Path) -> dict[str, Any]:
    config = {
        **DEFAULT_CONFIG,
        "ollama": dict(DEFAULT_CONFIG["ollama"]),
        "preprocessing": dict(DEFAULT_CONFIG["preprocessing"]),
        "classification": dict(DEFAULT_CONFIG["classification"]),
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
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

    return config


def list_images(source: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in source.glob("**/*")
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
        if img.mode != "RGB":
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


def extract_category(
    response_text: str,
    categories: list[str],
    *,
    min_confidence: float = 0.85,
    fallback_category: str = "uncategorized",
    require_confidence_format: bool = True,
) -> str:
    """
    Extrae la categoría y la confianza del texto de respuesta. Si la confianza es menor al umbral, retorna 'uncategorized'.
    """
    normalized_categories = {re.sub(r"\s+", " ", c.lower().strip()): c for c in categories}
    normalized_fallback = fallback_category.strip() or "uncategorized"

    # Buscar formato: categoria|porcentaje
    match = re.search(r"([a-zA-Z0-9 _-]+)\s*\|\s*(\d{1,3})", response_text.strip())
    if match:
        cat_raw, conf_raw = match.groups()
        try:
            confidence = int(conf_raw) / 100.0
        except Exception:
            confidence = 0.0
        normalized = cat_raw.lower().strip()
        if normalized in normalized_categories and confidence >= min_confidence:
            return normalized_categories[normalized]
        return normalized_fallback

    if require_confidence_format:
        return normalized_fallback

    # Fallback: comportamiento anterior
    normalized_response = response_text.lower().strip()
    normalized_response = re.sub(r"[^a-z0-9\s]", " ", normalized_response)
    normalized_response = re.sub(r"\s+", " ", normalized_response).strip()
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
    return normalized_fallback


def _build_effective_prompt(prompt_template: str, categories: list[str], fallback_category: str) -> str:
    category_list = ", ".join(categories)
    base_prompt = prompt_template.format(categories=category_list).strip()
    contract_instructions = (
        "\n\nOutput contract (strict): reply with exactly one line in this format: "
        "<category>|<confidence_0_to_100>. "
        f"If confidence is low, use {fallback_category}|<confidence>. "
        "Do not add explanations or extra text."
    )
    return f"{base_prompt}{contract_instructions}"


def classify_with_ollama(
    payload_b64: str,
    categories: list[str],
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    min_confidence: float = 0.85,
    fallback_category: str = "uncategorized",
    require_confidence_format: bool = True,
) -> tuple[str, str]:
    """Returns (category, raw_response) tuple. Raises OllamaError on connection issues."""
    prompt = _build_effective_prompt(prompt_template, categories, fallback_category)
    try:
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [payload_b64],
                "stream": False,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise OllamaError(f"Ollama no responde ({exc.__class__.__name__}): {exc}") from exc
    except requests.RequestException as exc:
        raise OllamaError(f"Error en API de Ollama: {exc}") from exc

    output = response.json().get("response", "")
    category = extract_category(
        str(output),
        categories,
        min_confidence=min_confidence,
        fallback_category=fallback_category,
        require_confidence_format=require_confidence_format,
    )
    return category, str(output).strip()


def _configure_error_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline_errors")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _load_history_index(history_log_path: Path) -> dict[str, str]:
    history_index: dict[str, str] = {}
    if not history_log_path.exists():
        return history_index

    with history_log_path.open("r", encoding="utf-8") as history_file:
        for raw_line in history_file:
            if "|" not in raw_line:
                continue
            filename, category = raw_line.split("|", maxsplit=1)
            normalized_name = filename.strip()
            normalized_category = category.strip()
            if normalized_name:
                history_index[normalized_name.casefold()] = normalized_category
    return history_index


def _append_history_entry(history_log_path: Path, filename: str, category: str) -> None:
    history_log_path.parent.mkdir(parents=True, exist_ok=True)
    with history_log_path.open("a", encoding="utf-8") as history_file:
        history_file.write(f"{filename} | {category}\n")


def postprocess_category(category: str, image_path: Path, categories: list[str]) -> str:
    """
    Postproceso simple: si la categoría es 'uncategorized', intentar heurística por nombre de archivo.
    """
    if category != "uncategorized":
        return category
    # Ejemplo: si el nombre del archivo contiene una categoría, usarla
    filename = image_path.stem.lower()
    for cat in categories:
        if cat.lower() in filename:
            return cat
    return category

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
    min_confidence: float = 0.85,
    fallback_category: str = "uncategorized",
    require_confidence_format: bool = True,
    history_log_path: Path | None = None,
) -> Summary:
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    output_categories = list(dict.fromkeys(categories + [fallback_category]))
    for category in output_categories:
        (output / category).mkdir(parents=True, exist_ok=True)

    errors_logger = _configure_error_logger(output / "errors.log")
    effective_history_log_path = history_log_path or (source.parent / "history.log")
    history_index = _load_history_index(effective_history_log_path)

    images = list_images(source)
    counts: dict[str, int] = {category: 0 for category in output_categories}
    total_original_size = 0
    total_payload_size = 0
    skipped_images = 0
    processed_images = 0

    started = time.perf_counter()


    for image_path in images:
        image_key = image_path.name.casefold()
        if image_key in history_index:
            skipped_images += 1
            print(
                f"{image_path.name} | skipped=history | category={history_index[image_key]} "
                f"| processing_seconds=0.00"
            )
            continue

        image_started = time.perf_counter()
        category = "uncategorized"
        raw_response = ""
        ollama_error: OllamaError | None = None
        try:
            payload_b64, original_size, payload_size = preprocess_image_for_inference(
                image_path,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
            )
            try:
                category, raw_response = classify_with_ollama(
                    payload_b64,
                    categories,
                    prompt_template,
                    ollama_url,
                    model,
                    timeout_seconds,
                    min_confidence=min_confidence,
                    fallback_category=fallback_category,
                    require_confidence_format=require_confidence_format,
                )
            except OllamaError as exc:
                ollama_error = exc
                errors_logger.error("No response from Ollama for %s: %s", image_path, exc)
            except Exception as exc:
                errors_logger.error("Failed to classify %s: %s", image_path, exc)
        except Exception as exc:
            errors_logger.error("Failed to preprocess %s: %s", image_path, exc)
            original_size = image_path.stat().st_size
            payload_size = 0

        total_original_size += original_size
        total_payload_size += payload_size

        # Postproceso para mejorar la calidad de la clasificación
        category = postprocess_category(category, image_path, categories)

        if category not in output_categories:
            category = fallback_category

        destination = output / category / image_path.name
        image_seconds = time.perf_counter() - image_started

        if ollama_error:
            print(
                f"{image_path.name} | error=ollama | Sin respuesta de Ollama, abre tu terminal WSL"
            )
        else:
            print(
                f"{image_path.name} | original_size={original_size} bytes "
                f"| payload_size={payload_size} bytes | category={category} "
                f"| gemma4_response={raw_response} "
                f"| processing_seconds={image_seconds:.2f} "
                f"| destination={destination}"
            )

        counts[category] += 1
        processed_images += 1
        if dry_run:
            continue

        if operation == "move":
            shutil.move(str(image_path), str(destination))
        else:
            shutil.copy2(image_path, destination)

        if not ollama_error:
            _append_history_entry(effective_history_log_path, image_path.name, category)
            history_index[image_key] = category

    total_seconds = time.perf_counter() - started
    total_images = processed_images

    avg_original_size = (total_original_size / total_images) if total_images else 0.0
    avg_payload_size = (total_payload_size / total_images) if total_images else 0.0

    return Summary(
        counts=counts,
        total_images=total_images,
        skipped_images=skipped_images,
        total_seconds=total_seconds,
        avg_original_size=avg_original_size,
        avg_payload_size=avg_payload_size,
    )
