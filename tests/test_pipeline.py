import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from pipeline import (
    OllamaError,
    describe_with_ollama,
    extract_description,
    preprocess_image_for_inference,
    process_images,
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


def test_extract_description_with_json_and_tags() -> None:
    response = '{"description":"Retrato de una mujer sonriendo","tags":["retrato","mujer","sonrisa"]}'
    description, tags = extract_description(response)
    assert description == "Retrato de una mujer sonriendo"
    assert tags == ["retrato", "mujer", "sonrisa"]


def test_extract_description_fallback_when_invalid_json() -> None:
    description, tags = extract_description(
        "no-json",
        fallback_text="sin descripcion",
        require_json=True,
    )
    assert description == "sin descripcion"
    assert tags == []

    description2, tags2 = extract_description(
        "Un gato en el sofá",
        fallback_text="sin descripcion",
        require_json=False,
    )
    assert description2 == "Un gato en el sofá"
    assert tags2 == []


def test_describe_with_ollama_enforces_contract_and_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": '{"description":"Retrato de una mujer","tags":["retrato","mujer"]}'}

    def _fake_post(url: str, json: dict[str, object], timeout: int):
        captured_payload["url"] = url
        captured_payload["json"] = json
        captured_payload["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("pipeline.requests.post", _fake_post)

    description, tags, raw = describe_with_ollama(
        payload_b64="abc",
        prompt_template="Describe la imagen",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=10,
        fallback_text="sin descripcion",
        require_json=True,
    )

    assert description == "Retrato de una mujer"
    assert tags == ["retrato", "mujer"]
    assert raw == '{"description":"Retrato de una mujer","tags":["retrato","mujer"]}'
    assert captured_payload["url"] == "http://localhost:11434/api/generate"
    assert captured_payload["timeout"] == 10
    request_json = captured_payload["json"]
    assert isinstance(request_json, dict)
    assert request_json["model"] == "gemma4"
    assert request_json["images"] == ["abc"]
    assert request_json["options"] == {"temperature": 0}
    assert "Output contract (strict)" in str(request_json["prompt"])


def test_describe_with_ollama_raises_ollama_error_on_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    def _fake_post(*args, **kwargs):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr("pipeline.requests.post", _fake_post)

    with pytest.raises(OllamaError, match="Ollama no responde"):
        describe_with_ollama(
            payload_b64="abc",
            prompt_template="Test",
            ollama_url="http://localhost:11434",
            model="gemma4",
            timeout_seconds=10,
        )


def test_describe_with_ollama_raises_ollama_error_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    def _fake_post(*args, **kwargs):
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr("pipeline.requests.post", _fake_post)

    with pytest.raises(OllamaError, match="Ollama no responde"):
        describe_with_ollama(
            payload_b64="abc",
            prompt_template="Test",
            ollama_url="http://localhost:11434",
            model="gemma4",
            timeout_seconds=10,
        )


def test_process_images_dry_run_keeps_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    description_log = tmp_path / "descriptions.jsonl"
    source.mkdir()

    image_path = source / "img.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    monkeypatch.setattr(
        "pipeline.describe_with_ollama",
        lambda *args, **kwargs: ("Un gato en el sofá", ["gato"], '{"description":"Un gato","tags":["gato"]}'),
    )

    summary = process_images(
        source=source,
        output=output,
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="Describe",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=True,
        fallback_text="sin descripcion",
        require_json=True,
        description_log_path=description_log,
    )

    assert image_path.exists()
    assert not (output / "img.jpg").exists()
    assert not description_log.exists()
    assert summary.total_images == 1


def test_process_images_description_failure_uses_fallback_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    description_log = tmp_path / "descriptions.jsonl"
    source.mkdir()

    image_path = source / "img.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    def _raise(*args, **kwargs):
        raise RuntimeError("description failed")

    monkeypatch.setattr("pipeline.describe_with_ollama", _raise)

    summary = process_images(
        source=source,
        output=output,
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="Describe",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=False,
        fallback_text="sin descripcion",
        require_json=True,
        description_log_path=description_log,
    )

    assert (output / "img.jpg").exists()
    assert summary.total_images == 1
    assert "description failed" in (output / "errors.log").read_text(encoding="utf-8")
    record = json.loads(description_log.read_text(encoding="utf-8").splitlines()[0])
    assert record["description"] == "sin descripcion"


def test_process_images_skips_files_from_description_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    description_log = tmp_path / "descriptions.jsonl"
    source.mkdir()

    already_done = source / "done.jpg"
    _create_image(already_done, mode="RGB", size=(120, 120))
    description_log.write_text(
        json.dumps({"relative_path": "done.jpg", "description": "Ya descrita"}) + "\n",
        encoding="utf-8",
    )

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("describe_with_ollama should not run for files in history")

    monkeypatch.setattr("pipeline.describe_with_ollama", _should_not_be_called)

    summary = process_images(
        source=source,
        output=output,
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="Describe",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=False,
        fallback_text="sin descripcion",
        require_json=True,
        description_log_path=description_log,
    )

    assert summary.total_images == 0
    assert summary.skipped_images == 1
    assert not (output / "done.jpg").exists()


def test_process_images_appends_to_description_log_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    description_log = tmp_path / "descriptions.jsonl"
    source.mkdir()

    image_path = source / "new.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    monkeypatch.setattr(
        "pipeline.describe_with_ollama",
        lambda *args, **kwargs: ("Foto familiar", ["familia"], '{"description":"Foto familiar","tags":["familia"]}'),
    )

    summary = process_images(
        source=source,
        output=output,
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="Describe",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=False,
        fallback_text="sin descripcion",
        require_json=True,
        description_log_path=description_log,
    )

    assert summary.total_images == 1
    assert summary.skipped_images == 0
    record = json.loads(description_log.read_text(encoding="utf-8").splitlines()[0])
    assert record["description"] == "Foto familiar"
    assert record["tags"] == ["familia"]


def test_process_images_does_not_add_to_description_log_on_ollama_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    description_log = tmp_path / "descriptions.jsonl"
    source.mkdir()

    image_path = source / "failing.jpg"
    _create_image(image_path, mode="RGB", size=(120, 120))

    def _raise_ollama_error(*args, **kwargs):
        raise OllamaError("Connection refused: Ollama not running")

    monkeypatch.setattr("pipeline.describe_with_ollama", _raise_ollama_error)

    summary = process_images(
        source=source,
        output=output,
        operation="copy",
        max_side=1024,
        jpeg_quality=85,
        prompt_template="Describe",
        ollama_url="http://localhost:11434",
        model="gemma4",
        timeout_seconds=2,
        dry_run=False,
        fallback_text="sin descripcion",
        require_json=True,
        description_log_path=description_log,
    )

    assert summary.total_images == 1
    history_content = description_log.read_text(encoding="utf-8") if description_log.exists() else ""
    assert history_content == ""
    captured = capsys.readouterr()
    assert "Sin respuesta de Ollama, abre tu terminal WSL" in captured.out
    assert "error=ollama" in captured.out
