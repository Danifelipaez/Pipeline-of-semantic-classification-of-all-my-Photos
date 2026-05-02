from __future__ import annotations

import base64
import json
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
    "ollama": {
        "url": "http://localhost:11434",
        "model": "gemma4",
        "timeout_seconds": 120,
    },
    "description_prompt": (
        "Eres un modelo de visión que crea descripciones semánticas útiles para búsqueda. "
        "Describe la escena en español en 1 o 2 frases y genera una lista corta de etiquetas."
    ),
    "description": {
        "fallback_text": "descripcion no disponible",
        "require_json": True,
    },
    "description_output": "descriptions.jsonl",
    "preprocessing": {
        "max_side": 1024,
        "jpeg_quality": 85,
    },
}


@dataclass
class Summary:
    total_images: int
    skipped_images: int
    total_seconds: float
    avg_original_size: float
    avg_payload_size: float
    description_log_path: Path


def load_config(config_path: Path) -> dict[str, Any]:
    config = {
        **DEFAULT_CONFIG,
        "ollama": dict(DEFAULT_CONFIG["ollama"]),
        "preprocessing": dict(DEFAULT_CONFIG["preprocessing"]),
        "description": dict(DEFAULT_CONFIG["description"]),
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for key, value in loaded.items():
            if key in {"ollama", "preprocessing", "classification"} and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    operation = str(config.get("operation", "copy")).lower()
    if operation not in {"copy", "move"}:
        raise ValueError("operation must be 'copy' or 'move'")
    config["operation"] = operation

    fallback_text = str(config["description"].get("fallback_text", "descripcion no disponible")).strip()
    if not fallback_text:
        raise ValueError("description.fallback_text cannot be empty")
    config["description"]["fallback_text"] = fallback_text
    config["description"]["require_json"] = bool(config["description"].get("require_json", True))

    description_output = str(config.get("description_output", "descriptions.jsonl")).strip()
    if not description_output:
        raise ValueError("description_output cannot be empty")
    config["description_output"] = description_output

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


def _extract_json_payload(response_text: str) -> dict[str, Any] | None:
    if not response_text:
        return None
    raw = response_text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    end_inclusive = end + 1
    candidate = raw[start:end_inclusive]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_tags(tags: object) -> list[str]:
    if isinstance(tags, str):
        parts = re.split(r"[,;]", tags)
        cleaned = [part.strip() for part in parts if part.strip()]
        return cleaned
    if isinstance(tags, list):
        cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
        return cleaned
    return []


def extract_description(
    response_text: str,
    *,
    fallback_text: str = "descripcion no disponible",
    require_json: bool = True,
) -> tuple[str, list[str]]:
    parsed = _extract_json_payload(response_text)
    if parsed is not None:
        description = str(parsed.get("description", "")).strip()
        tags = _normalize_tags(parsed.get("tags", []))
        if description:
            return description, tags
        return fallback_text, tags
    if require_json:
        return fallback_text, []
    description = response_text.strip()
    return (description if description else fallback_text), []


def _build_effective_prompt(prompt_template: str) -> str:
    base_prompt = prompt_template.strip()
    contract_instructions = (
        "\n\nOutput contract (strict): return a single JSON object in one line with keys "
        '"description" (string) and "tags" (array of strings). '
        "Do not add explanations or extra text."
    )
    return f"{base_prompt}{contract_instructions}"


def describe_with_ollama(
    payload_b64: str,
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    fallback_text: str = "descripcion no disponible",
    require_json: bool = True,
) -> tuple[str, list[str], str]:
    """Returns (description, tags, raw_response) tuple. Raises OllamaError on connection issues."""
    prompt = _build_effective_prompt(prompt_template)
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
    description, tags = extract_description(
        str(output),
        fallback_text=fallback_text,
        require_json=require_json,
    )
    return description, tags, str(output).strip()


def _configure_error_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline_errors")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _load_description_index(description_log_path: Path) -> set[str]:
    description_index: set[str] = set()
    if not description_log_path.exists():
        return description_index

    with description_log_path.open("r", encoding="utf-8") as description_file:
        for raw_line in description_file:
            line = raw_line.strip()
            if not line:
                continue
            if line.lstrip().startswith("{"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(record.get("relative_path") or record.get("filename") or "").strip()
                if key:
                    description_index.add(key.casefold())
                continue
            if "|" in line:
                filename, _ = line.split("|", maxsplit=1)
                normalized_name = filename.strip()
                if normalized_name:
                    description_index.add(normalized_name.casefold())
    return description_index


def _append_description_entry(description_log_path: Path, record: dict[str, Any]) -> None:
    description_log_path.parent.mkdir(parents=True, exist_ok=True)
    with description_log_path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, ensure_ascii=False))
        history_file.write("\n")


def _description_keys(relative_path: Path, filename: str) -> tuple[str, str]:
    """Return keys for relative path and filename to support legacy log lookups."""
    return str(relative_path).casefold(), filename.casefold()

def process_images(
    *,
    source: Path,
    output: Path,
    operation: str,
    max_side: int,
    jpeg_quality: int,
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    dry_run: bool,
    fallback_text: str = "descripcion no disponible",
    require_json: bool = True,
    description_log_path: Path | None = None,
) -> Summary:
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    errors_logger = _configure_error_logger(output / "errors.log")
    effective_description_log_path = description_log_path or (output / "descriptions.jsonl")
    description_index = _load_description_index(effective_description_log_path)

    images = list_images(source)
    total_original_size = 0
    total_payload_size = 0
    skipped_images = 0
    processed_images = 0

    started = time.perf_counter()


    for image_path in images:
        relative_path = image_path.relative_to(source)
        image_key, filename_key = _description_keys(relative_path, image_path.name)
        if image_key in description_index or filename_key in description_index:
            skipped_images += 1
            print(
                f"{image_path.name} | skipped=description_log | processing_seconds=0.00"
            )
            continue

        image_started = time.perf_counter()
        description = fallback_text
        tags: list[str] = []
        raw_response = ""
        ollama_error: OllamaError | None = None
        try:
            payload_b64, original_size, payload_size = preprocess_image_for_inference(
                image_path,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
            )
            try:
                description, tags, raw_response = describe_with_ollama(
                    payload_b64,
                    prompt_template,
                    ollama_url,
                    model,
                    timeout_seconds,
                    fallback_text=fallback_text,
                    require_json=require_json,
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

        destination = output / relative_path
        image_seconds = time.perf_counter() - image_started

        if ollama_error:
            print(
                f"{image_path.name} | error=ollama | Sin respuesta de Ollama, abre tu terminal WSL"
            )
        else:
            print(
                f"{image_path.name} | original_size={original_size} bytes "
                f"| payload_size={payload_size} bytes | description={description} "
                f"| tags={', '.join(tags)} | gemma4_response={raw_response} "
                f"| processing_seconds={image_seconds:.2f} "
                f"| destination={destination}"
            )

        processed_images += 1
        if dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if operation == "move":
            shutil.move(str(image_path), str(destination))
        else:
            shutil.copy2(image_path, destination)

        if not ollama_error:
            _append_description_entry(
                effective_description_log_path,
                {
                    "filename": image_path.name,
                    "relative_path": str(relative_path),
                    "source_path": str(image_path),
                    "output_path": str(destination),
                    "description": description,
                    "tags": tags,
                    "raw_response": raw_response,
                },
            )
            description_index.update({image_key, filename_key})

    total_seconds = time.perf_counter() - started
    total_images = processed_images

    avg_original_size = (total_original_size / total_images) if total_images else 0.0
    avg_payload_size = (total_payload_size / total_images) if total_images else 0.0

    return Summary(
        total_images=total_images,
        skipped_images=skipped_images,
        total_seconds=total_seconds,
        avg_original_size=avg_original_size,
        avg_payload_size=avg_payload_size,
        description_log_path=effective_description_log_path,
    )
