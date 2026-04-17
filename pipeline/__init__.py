from __future__ import annotations

from pathlib import Path

from pipeline.application.processor import process_images as _process_images
from pipeline.domain.classification import extract_category, postprocess_category
from pipeline.domain.config import DEFAULT_CONFIG, load_config
from pipeline.domain.contracts import OllamaError, Summary
from pipeline.domain.workers import resolve_classification_workers
from pipeline.infrastructure import ollama_client as _ollama_client
from pipeline.infrastructure.image_processing import preprocess_image_for_inference

requests = _ollama_client.requests


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
    """Compatibility wrapper around infrastructure Ollama client."""
    return _ollama_client.classify_with_ollama(
        payload_b64=payload_b64,
        categories=categories,
        prompt_template=prompt_template,
        ollama_url=ollama_url,
        model=model,
        timeout_seconds=timeout_seconds,
        min_confidence=min_confidence,
        fallback_category=fallback_category,
        require_confidence_format=require_confidence_format,
    )


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
    classification_workers: int = 1,
    history_log_path: Path | None = None,
) -> Summary:
    """Compatibility wrapper preserving monkeypatch behavior in existing tests."""
    return _process_images(
        source=source,
        output=output,
        categories=categories,
        operation=operation,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        prompt_template=prompt_template,
        ollama_url=ollama_url,
        model=model,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
        min_confidence=min_confidence,
        fallback_category=fallback_category,
        require_confidence_format=require_confidence_format,
        classification_workers=classification_workers,
        history_log_path=history_log_path,
        classify_image_fn=classify_with_ollama,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "OllamaError",
    "Summary",
    "classify_with_ollama",
    "extract_category",
    "load_config",
    "postprocess_category",
    "preprocess_image_for_inference",
    "process_images",
    "requests",
    "resolve_classification_workers",
]
