"""Pipeline package for semantic classification of photos."""

from .core import (
    DEFAULT_CONFIG,
    OllamaError,
    SUPPORTED_EXTENSIONS,
    Summary,
    describe_with_ollama,
    extract_description,
    list_images,
    load_config,
    preprocess_image_for_inference,
    process_images,
)

__all__ = [
    "DEFAULT_CONFIG",
    "OllamaError",
    "SUPPORTED_EXTENSIONS",
    "Summary",
    "describe_with_ollama",
    "extract_description",
    "list_images",
    "load_config",
    "preprocess_image_for_inference",
    "process_images",
]
