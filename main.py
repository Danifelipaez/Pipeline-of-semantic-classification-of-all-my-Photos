from pathlib import Path

import json
import typer

from pipeline import load_config
from pipeline.application.ingest_service import ingest_and_index
from pipeline.application.index_service import IndexService
from pipeline.infrastructure.ollama_client import OllamaClient
from pipeline.application.search_service import SearchService
from pipeline.domain.models import PhotoRecord


def main(
    source: Path | None = typer.Option(None, help="Path to source photos folder"),
    output: Path | None = typer.Option(None, help="Path to output folder"),
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Classify without moving/copying files"),
) -> None:
    loaded = load_config(config)

    source_path = Path(source) if source is not None else Path(loaded["source"])
    output_path = Path(output) if output is not None else Path(loaded["output"])
    description_log_path = Path(loaded["description_output"])
    if not description_log_path.is_absolute():
        description_log_path = output_path / description_log_path

    summary = process_images(
        source=source_path,
        output=output_path,
        operation=loaded["operation"],
        max_side=int(loaded["preprocessing"]["max_side"]),
        jpeg_quality=int(loaded["preprocessing"]["jpeg_quality"]),
        prompt_template=str(loaded["description_prompt"]),
        ollama_url=str(loaded["ollama"]["url"]),
        model=str(loaded["ollama"]["model"]),
        timeout_seconds=int(loaded["ollama"]["timeout_seconds"]),
        fallback_text=str(loaded["description"]["fallback_text"]),
        require_json=bool(loaded["description"]["require_json"]),
        dry_run=dry_run,
        description_log_path=description_log_path,
    )

    typer.echo("\nSummary:")
    typer.echo(f"Total images: {summary.total_images}")
    typer.echo(f"Skipped (already in description log): {summary.skipped_images}")
    typer.echo(f"Total processing time (s): {summary.total_seconds:.2f}")
    typer.echo(f"Average original size (bytes): {summary.avg_original_size:.2f}")
    typer.echo(f"Average inference payload size (bytes): {summary.avg_payload_size:.2f}")
    typer.echo(f"Description log: {summary.description_log_path}")
    # after processing, index into SQLite
    db_path = output_path / "index.db"
    ingest_and_index(
        source=source_path,
        output=output_path,
        operation=loaded["operation"],
        max_side=int(loaded["preprocessing"]["max_side"]),
        jpeg_quality=int(loaded["preprocessing"]["jpeg_quality"]),
        prompt_template=str(loaded["description_prompt"]),
        ollama_url=str(loaded["ollama"]["url"]),
        model=str(loaded["ollama"]["model"]),
        timeout_seconds=int(loaded["ollama"]["timeout_seconds"]),
        fallback_text=str(loaded["description"]["fallback_text"]),
        require_json=bool(loaded["description"]["require_json"]),
        dry_run=dry_run,
        description_log_path=description_log_path,
        db_path=db_path,
    )


def rebuild_index(
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
):
    loaded = load_config(config)
    output_path = Path(loaded["output"]) if loaded.get("output") else Path(loaded["output"])
    db_path = output_path / "index.db"
    ollama = OllamaClient(loaded["ollama"]["url"], loaded["ollama"]["model"], int(loaded["ollama"]["timeout_seconds"]))
    index = IndexService(db_path)
    # rebuild by reading descriptions.jsonl
    description_log = Path(loaded.get("description_output", "descriptions.jsonl"))
    if not description_log.is_absolute():
        description_log = output_path / description_log
    if not description_log.exists():
        typer.echo("No description log found to index.")
        raise typer.Exit(code=1)

    records = []
    with description_log.open("r", encoding="utf-8") as f:
        for raw in f:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            rec = PhotoRecord(
                relative_path=obj.get("relative_path") or obj.get("filename"),
                filename=obj.get("filename"),
                source_path=obj.get("source_path"),
                output_path=obj.get("output_path"),
                description=obj.get("description", ""),
                tags=obj.get("tags", []),
                entities=obj.get("entities", {}),
                scene_attributes=obj.get("scene_attributes", {}),
                raw_response=obj.get("raw_response", ""),
            )
            records.append(rec)

    index.index_photos(records)
    typer.echo(f"Indexed {len(records)} records into {db_path}")


def search(
    query: str = typer.Argument(..., help="Consulta en lenguaje natural"),
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
    semantic: bool = typer.Option(False, help="Usar ranking semántico con Gemma4"),
    limit: int = typer.Option(20, help="Límite de resultados"),
):
    loaded = load_config(config)
    output_path = Path(loaded["output"]) if loaded.get("output") else Path(loaded["output"])
    db_path = output_path / "index.db"
    ollama = OllamaClient(loaded["ollama"]["url"], loaded["ollama"]["model"], int(loaded["ollama"]["timeout_seconds"]))
    searchsvc = SearchService(db_path, ollama)
    candidates = searchsvc.text_search(query, limit=limit)
    if semantic and candidates:
        ranked = searchsvc.semantic_rank(query, candidates)
        for rowid, path, score in ranked[:limit]:
            typer.echo(f"{path} | score={score}")
    else:
        for rowid, path, desc in candidates[:limit]:
            typer.echo(f"{path} | {desc}")


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(main)
    app.command()(rebuild_index)
    app.command()(search)
    app()
