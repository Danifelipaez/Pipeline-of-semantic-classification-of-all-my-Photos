from __future__ import annotations

from dataclasses import dataclass


class OllamaError(Exception):
    """Raised when Ollama API is unreachable or fails to respond."""


@dataclass
class Summary:
    counts: dict[str, int]
    total_images: int
    skipped_images: int
    total_seconds: float
    avg_original_size: float
    avg_payload_size: float
