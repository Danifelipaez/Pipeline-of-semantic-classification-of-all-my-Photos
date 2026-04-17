from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


def preprocess_image_for_inference(
    image_path: Path,
    *,
    max_side: int,
    jpeg_quality: int,
) -> tuple[str, int, int]:
    """Resize and encode image to JPEG base64 payload for model inference."""
    original_size = image_path.stat().st_size

    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        longest_side = max(width, height)
        if longest_side > max_side:
            scale = max_side / float(longest_side)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        payload_bytes = buffer.getvalue()

    payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
    return payload_b64, original_size, len(payload_bytes)
