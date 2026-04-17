from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(source: Path) -> list[Path]:
    """Return sorted image paths recursively from source directory."""
    return sorted(
        [
            path
            for path in source.glob("**/*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
