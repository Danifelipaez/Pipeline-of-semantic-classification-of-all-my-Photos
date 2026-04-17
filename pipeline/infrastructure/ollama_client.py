from __future__ import annotations

import requests

from pipeline.domain.classification import build_effective_prompt, extract_category
from pipeline.domain.contracts import OllamaError


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
    """Run classification request against Ollama and return category + raw response."""
    prompt = build_effective_prompt(prompt_template, categories, fallback_category)

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
