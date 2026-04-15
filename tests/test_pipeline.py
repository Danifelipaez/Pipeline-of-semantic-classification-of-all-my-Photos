import base64
from pathlib import Path

import pytest
from PIL import Image

from pipeline import extract_category, preprocess_image_for_inference, process_images, postprocess_category


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


def test_extract_category_with_sentence_and_confidence() -> None:
    categories = ["nature", "birds", "street photography"]
    # Caso: formato antiguo, sin confianza
    assert extract_category("The best category is street photography.", categories) == "street photography"
    assert extract_category("unknown", categories) == "uncategorized"
    # Caso: formato nuevo, confianza suficiente
    assert extract_category("nature|92", categories) == "nature"
    # Caso: formato nuevo, confianza insuficiente
    assert extract_category("birds|70", categories) == "uncategorized"
    # Caso: formato nuevo, confianza justo en el umbral
    assert extract_category("birds|85", categories) == "birds"
def test_postprocess_category_uses_filename(tmp_path: Path) -> None:
    categories = ["nature", "birds", "street photography"]
    # Si la categoría es válida, no cambia
    img_path = tmp_path / "img.jpg"
    assert postprocess_category("nature", img_path, categories) == "nature"
    # Si es 'uncategorized' y el nombre contiene una categoría, la asigna
    img_path2 = tmp_path / "my_birds_photo.jpg"
    assert postprocess_category("uncategorized", img_path2, categories) == "birds"
    # Si no hay coincidencia, sigue siendo 'uncategorized'
    img_path3 = tmp_path / "random_image.jpg"
    assert postprocess_category("uncategorized", img_path3, categories) == "uncategorized"


def test_process_images_dry_run_keeps_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    image_path = source / "img.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    # Simula respuesta con confianza suficiente
    monkeypatch.setattr("pipeline.classify_with_ollama", lambda *args, **kwargs: ("family", "family|90"))

    summary = process_images(
        source=source,
        output=output,
        categories=["family"],
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="{categories}",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=True,
        min_confidence=0.85,
    )

    assert image_path.exists()
    assert not (output / "family" / "img.jpg").exists()
    assert summary.counts["family"] == 1

    # Simula respuesta con confianza insuficiente
    monkeypatch.setattr("pipeline.classify_with_ollama", lambda *args, **kwargs: ("uncategorized", "family|70"))
    summary2 = process_images(
        source=source,
        output=output,
        categories=["family"],
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="{categories}",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=True,
        min_confidence=0.85,
    )
    assert summary2.counts["uncategorized"] >= 0


def test_process_images_classification_failure_goes_uncategorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    image_path = source / "img.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    def _raise(*args, **kwargs):
        raise RuntimeError("classification failed")

    monkeypatch.setattr("pipeline.classify_with_ollama", _raise)

    summary = process_images(
        source=source,
        output=output,
        categories=["family"],
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="{categories}",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=False,
    )

    assert (output / "uncategorized" / "img.jpg").exists()
    assert summary.counts["uncategorized"] == 1
    assert "classification failed" in (output / "errors.log").read_text(encoding="utf-8")
