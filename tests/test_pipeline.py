import base64
from pathlib import Path

from PIL import Image

from pipeline import extract_category, preprocess_image_for_inference, process_images


def _create_image(path: Path, mode: str = "RGB", size: tuple[int, int] = (2000, 1200)) -> None:
    color = (255, 0, 0, 255) if mode == "RGBA" else (255, 0, 0)
    image = Image.new(mode, size, color)
    image.save(path)


def test_preprocess_resizes_and_encodes_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.png"
    _create_image(image_path, mode="RGBA")

    payload_b64, original_size, payload_size = preprocess_image_for_inference(
        image_path,
        max_side=1024,
        jpeg_quality=85,
    )

    assert original_size == image_path.stat().st_size
    assert payload_size > 0

    decoded = base64.b64decode(payload_b64)
    assert decoded[:2] == b"\xff\xd8"


def test_extract_category_with_sentence() -> None:
    categories = ["nature", "birds", "street photography"]
    assert extract_category("The best category is street photography.", categories) == "street photography"
    assert extract_category("unknown", categories) == "uncategorized"


def test_process_images_dry_run_keeps_source_files(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    image_path = source / "img.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    monkeypatch.setattr("pipeline.classify_with_ollama", lambda *args, **kwargs: "family")

    summary = process_images(
        source=source,
        output=output,
        categories=["family"],
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="{categories}",
        ollama_url="http://localhost:11434",
        model="gemma3",
        timeout_seconds=2,
        dry_run=True,
    )

    assert image_path.exists()
    assert not (output / "family" / "img.jpg").exists()
    assert summary.counts["family"] == 1
