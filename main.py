from pathlib import Path

import typer

from pipeline import load_config, process_images


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


if __name__ == "__main__":
    typer.run(main)
