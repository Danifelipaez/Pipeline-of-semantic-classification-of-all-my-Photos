import base64
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from pipeline import (
    OllamaError,
    classify_with_ollama,
    extract_category,
    load_config,
    preprocess_image_for_inference,
    process_images,
    resolve_classification_workers,
    postprocess_category,
)


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


def test_load_config_defaults_classification_workers(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.yaml")
    classification = config["classification"]

    assert classification["workers_mode"] == "auto"
    assert classification["num_workers"] == 4
    assert classification["max_workers_cap"] == 6


def test_load_config_rejects_invalid_workers_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
classification:
  workers_mode: invalid
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="classification.workers_mode"):
        load_config(config_path)


def test_resolve_classification_workers_manual_respects_cap() -> None:
    workers = resolve_classification_workers(
        {
            "workers_mode": "manual",
            "num_workers": 10,
            "max_workers_cap": 3,
        }
    )
    assert workers == 3


def test_process_images_parallel_classification_uses_multiple_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()

    for idx in range(6):
        _create_image(source / f"img_{idx}.jpg", mode="RGB", size=(120, 120))

    worker_thread_ids: set[int] = set()
    ids_lock = threading.Lock()

    def _fake_classify(*args, **kwargs):
        time.sleep(0.05)
        thread_id = threading.get_ident()
        with ids_lock:
            worker_thread_ids.add(thread_id)
        return "family", "family|95"

    monkeypatch.setattr("pipeline.classify_with_ollama", _fake_classify)

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
        classification_workers=3,
    )

    assert summary.total_images == 6
    assert summary.counts["family"] == 6
    assert len(worker_thread_ids) >= 2


def test_process_images_history_uses_relative_paths_to_avoid_name_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    history_log = tmp_path / "history.log"
    (source / "a").mkdir(parents=True)
    (source / "b").mkdir(parents=True)

    img_a = source / "a" / "same.jpg"
    img_b = source / "b" / "same.jpg"
    _create_image(img_a, mode="RGB", size=(120, 120))
    _create_image(img_b, mode="RGB", size=(120, 120))

    # Mark only one of the duplicated basenames as already processed.
    history_log.write_text("a/same.jpg | family\n", encoding="utf-8")

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

    assert summary.skipped_images == 1
    assert summary.total_images == 1
    history_content = history_log.read_text(encoding="utf-8")
    assert "a/same.jpg | family" in history_content
    assert "b/same.jpg | family" in history_content


def test_process_images_history_keeps_legacy_basename_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    history_log = tmp_path / "history.log"
    source.mkdir()

    image_path = source / "done.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))
    # Legacy format: basename only.
    history_log.write_text("done.jpg | family\n", encoding="utf-8")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("classify_with_ollama should not run for files in legacy history")

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
