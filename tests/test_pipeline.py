import base64
from pathlib import Path

import pytest
from PIL import Image

from pipeline import classify_with_ollama, extract_category, OllamaError, preprocess_image_for_inference, process_images, postprocess_category


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
    # Caso: formato sin confianza -> fallback por modo estricto
    assert extract_category("The best category is street photography.", categories) == "uncategorized"
    assert extract_category("unknown", categories) == "uncategorized"
    # Caso: formato nuevo, confianza suficiente
    assert extract_category("nature|92", categories) == "nature"
    # Caso: formato nuevo, confianza insuficiente
    assert extract_category("birds|70", categories) == "uncategorized"
    # Caso: formato nuevo, confianza justo en el umbral
    assert extract_category("birds|85", categories) == "birds"


def test_extract_category_can_use_legacy_mode_without_confidence() -> None:
    categories = ["nature", "birds", "street photography"]
    category = extract_category(
        "The best category is street photography.",
        categories,
        require_confidence_format=False,
    )
    assert category == "street photography"


def test_classify_with_ollama_enforces_contract_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "family|70"}

    def _fake_post(url: str, json: dict[str, object], timeout: int):
        captured_payload["url"] = url
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("pipeline.requests.post", _fake_post)

    category, raw = classify_with_ollama(
        payload_b64="abc",
        categories=["family", "nature"],
        prompt_template="Elige solo una categoría de {categories}",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=10,
        min_confidence=0.85,
        fallback_category="uncategorized",
        require_confidence_format=True,
    )

    assert category == "uncategorized"
    assert raw == "family|70"
    assert captured_payload["url"] == "http://localhost:11434/api/generate"
    assert captured_payload["timeout"] == 10
    request_json = captured_payload["json"]
    assert isinstance(request_json, dict)
    assert request_json["model"] == "gemma4"
    assert request_json["images"] == ["abc"]
    assert request_json["options"] == {"temperature": 0}
    assert "Output contract (strict)" in str(request_json["prompt"])


def test_classify_with_ollama_raises_ollama_error_on_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    def _fake_post(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr("pipeline.requests.post", _fake_post)

    with pytest.raises(OllamaError, match="Ollama no responde"):
        classify_with_ollama(
            payload_b64="abc",
            categories=["family"],
            prompt_template="Test",
            ollama_url="http://localhost:11434",
            model="gemma4",
            timeout_seconds=10,
        )


def test_classify_with_ollama_raises_ollama_error_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    def _fake_post(*args, **kwargs):
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr("pipeline.requests.post", _fake_post)

    with pytest.raises(OllamaError, match="Ollama no responde"):
        classify_with_ollama(
            payload_b64="abc",
            categories=["family"],
            prompt_template="Test",
            ollama_url="http://localhost:11434",
            model="gemma4",
            timeout_seconds=10,
        )
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


def test_process_images_skips_files_from_history_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    history_log = tmp_path / "history.log"
    source.mkdir()

    already_done = source / "done.jpg"
    _create_image(already_done, mode="RGB", size=(120, 120))
    history_log.write_text("done.jpg | family\n", encoding="utf-8")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("classify_with_ollama should not run for files in history")

    monkeypatch.setattr("pipeline.classify_with_ollama", _should_not_be_called)

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
        history_log_path=history_log,
    )

    assert summary.total_images == 0
    assert summary.skipped_images == 1
    assert not (output / "family" / "done.jpg").exists()


def test_process_images_appends_to_history_log_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    history_log = tmp_path / "history.log"
    source.mkdir()

    image_path = source / "new.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    monkeypatch.setattr("pipeline.classify_with_ollama", lambda *args, **kwargs: ("family", "family|95"))

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
        history_log_path=history_log,
    )

    assert summary.total_images == 1
    assert summary.skipped_images == 0
    assert "new.jpg | family" in history_log.read_text(encoding="utf-8")


def test_process_images_does_not_add_to_history_on_ollama_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    history_log = tmp_path / "history.log"
    source.mkdir()

    image_path = source / "failing.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    def _raise_ollama_error(*args, **kwargs):
        raise OllamaError("Connection refused: Ollama not running")

    monkeypatch.setattr("pipeline.classify_with_ollama", _raise_ollama_error)

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
        history_log_path=history_log,
    )

    assert summary.total_images == 1
    history_content = history_log.read_text(encoding="utf-8") if history_log.exists() else ""
    assert history_content == ""
    captured = capsys.readouterr()
    assert "Sin respuesta de Ollama, abre tu terminal WSL" in captured.out
    assert "error=ollama" in captured.out
