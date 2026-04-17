from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import typer

from pipeline.domain.classification import postprocess_category
from pipeline.domain.contracts import OllamaError, Summary
from pipeline.infrastructure.history import (
    append_history_entry,
    history_key_for_image,
    load_history_index,
)
from pipeline.infrastructure.image_processing import preprocess_image_for_inference
from pipeline.infrastructure.logging_setup import configure_error_logger
from pipeline.infrastructure.ollama_client import classify_with_ollama
from pipeline.infrastructure.storage import list_images

ClassifierFn = Callable[
    [str, list[str], str, str, str, int, float, str, bool],
    tuple[str, str],
]


def _prepare_output_categories(output: Path, categories: list[str], fallback_category: str) -> list[str]:
    output_categories = list(dict.fromkeys(categories + [fallback_category]))
    for category in output_categories:
        (output / category).mkdir(parents=True, exist_ok=True)
    return output_categories


def _filter_pending_images(
    images: list[Path],
    source: Path,
    history_index: dict[str, str],
) -> tuple[list[Path], int]:
    images_to_process = [
        image
        for image in images
        if history_key_for_image(image, source) not in history_index and image.name.casefold() not in history_index
    ]
    skipped_images = len(images) - len(images_to_process)
    return images_to_process, skipped_images


def process_images(
    *,
    source: Path,
    output: Path,
    categories: list[str],
    operation: str,
    max_side: int,
    jpeg_quality: int,
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    dry_run: bool,
    min_confidence: float = 0.85,
    fallback_category: str = "uncategorized",
    require_confidence_format: bool = True,
    classification_workers: int = 1,
    history_log_path: Path | None = None,
    classify_image_fn: ClassifierFn | None = None,
) -> Summary:
    """Process images using parallel preprocessing and classification pools."""
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    output_categories = _prepare_output_categories(output, categories, fallback_category)

    errors_logger = configure_error_logger(output / "errors.log")
    effective_history_log_path = history_log_path or (source.parent / "history.log")

    typer.echo(f"Loading history from {effective_history_log_path}...")
    history_index = load_history_index(effective_history_log_path)
    typer.echo(f"  Loaded {len(history_index)} previously processed images from history")

    typer.echo(f"Scanning images in {source}...")
    images = list_images(source)
    typer.echo(f"  Found {len(images)} total images")

    images_to_process, skipped_images = _filter_pending_images(images, source, history_index)
    if skipped_images > 0:
        typer.echo(f"  Skipping {skipped_images} already processed images")

    if not images_to_process:
        typer.echo("No new images to process. Done!")
        return Summary(
            counts={category: 0 for category in output_categories},
            total_images=0,
            skipped_images=skipped_images,
            total_seconds=0.0,
            avg_original_size=0.0,
            avg_payload_size=0.0,
        )

    classify_workers = max(1, int(classification_workers))
    typer.echo(
        f"Processing {len(images_to_process)} new images "
        f"(prep workers=4, classify workers={classify_workers})..."
    )

    counts: dict[str, int] = {category: 0 for category in output_categories}
    total_original_size = 0
    total_payload_size = 0
    processed_images = 0

    started = time.perf_counter()
    prep_batch_size = 50
    prep_completed = 0
    classifier = classify_image_fn or classify_with_ollama

    prep_executor = ThreadPoolExecutor(max_workers=4)
    classify_executor = ThreadPoolExecutor(max_workers=classify_workers)
    cancelled = False

    try:
        typer.echo("\nClassifying and organizing images...")
        for batch_idx, batch_start in enumerate(range(0, len(images_to_process), prep_batch_size), 1):
            batch = images_to_process[batch_start:batch_start + prep_batch_size]
            batch_cache: dict[Path, tuple[str, int, int]] = {}
            batch_errors: dict[Path, Exception] = {}
            prep_futures: dict[Any, Path] = {}
            classify_futures: dict[Any, tuple[Path, float, float]] = {}
            classify_results: dict[Path, tuple[str, str, float, float]] = {}
            ollama_errors: dict[Path, OllamaError] = {}

            for image_path in batch:
                future = prep_executor.submit(
                    preprocess_image_for_inference,
                    image_path,
                    max_side=max_side,
                    jpeg_quality=jpeg_quality,
                )
                prep_futures[future] = image_path

            for future in as_completed(prep_futures):
                image_path = prep_futures[future]
                try:
                    payload_b64, original_size, payload_size = future.result()
                    batch_cache[image_path] = (payload_b64, original_size, payload_size)
                    prep_completed += 1
                    typer.echo(
                        f"[{prep_completed}/{len(images_to_process)}] "
                        f"Prepped: {image_path.name} ({payload_size} bytes)"
                    )
                except Exception as exc:
                    batch_errors[image_path] = exc
                    prep_completed += 1
                    typer.echo(
                        f"[{prep_completed}/{len(images_to_process)}] "
                        f"ERROR prep: {image_path.name} - {exc}"
                    )

            for image_path in batch:
                if image_path in batch_errors:
                    continue

                payload_b64, _, _ = batch_cache[image_path]
                classify_start_wall = time.time()
                classify_start_perf = time.perf_counter()
                future = classify_executor.submit(
                    classifier,
                    payload_b64,
                    categories,
                    prompt_template,
                    ollama_url,
                    model,
                    timeout_seconds,
                    min_confidence,
                    fallback_category,
                    require_confidence_format,
                )
                classify_futures[future] = (image_path, classify_start_wall, classify_start_perf)

            batch_positions = {image_path: batch_start + offset for offset, image_path in enumerate(batch, 1)}

            for future in as_completed(classify_futures):
                image_path, classify_start_wall, classify_start_perf = classify_futures[future]
                global_index = batch_positions[image_path]
                try:
                    category, raw_response = future.result()
                    classify_seconds = time.perf_counter() - classify_start_perf
                    display_category = postprocess_category(category, image_path, categories)
                    if display_category not in output_categories:
                        display_category = fallback_category
                    classify_results[image_path] = (
                        display_category,
                        raw_response,
                        classify_start_wall,
                        classify_seconds,
                    )

                    destination = output / display_category / image_path.name
                    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(classify_start_wall))
                    typer.echo(
                        f"\n[Batch {batch_idx}] [{global_index}/{len(images_to_process)}]\n"
                        f"  Start time: {start_time_str}\n"
                        f"  File: {image_path.name}\n"
                        f"  Category: {display_category}\n"
                        f"  Classify time: {classify_seconds:.3f}s\n"
                        f"  Gemma response: {raw_response.strip()}\n"
                        f"  Output path: {destination}"
                    )
                except OllamaError as exc:
                    ollama_errors[image_path] = exc
                    errors_logger.error("No response from Ollama for %s: %s", image_path, exc)
                    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(classify_start_wall))
                    destination = output / fallback_category / image_path.name
                    typer.echo(
                        f"\n[Batch {batch_idx}] [{global_index}/{len(images_to_process)}]\n"
                        f"  Start time: {start_time_str}\n"
                        f"  File: {image_path.name}\n"
                        f"  Status: error=ollama\n"
                        f"  Category: {fallback_category}\n"
                        f"  Classify time: N/A\n"
                        f"  Gemma response: ERROR - {exc}\n"
                        f"  Note: Sin respuesta de Ollama, abre tu terminal WSL\n"
                        f"  Output path: {destination}"
                    )
                except Exception as exc:
                    errors_logger.error("Failed to classify %s: %s", image_path, exc)
                    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(classify_start_wall))
                    destination = output / fallback_category / image_path.name
                    typer.echo(
                        f"\n[Batch {batch_idx}] [{global_index}/{len(images_to_process)}]\n"
                        f"  Start time: {start_time_str}\n"
                        f"  File: {image_path.name}\n"
                        f"  Status: error=classification\n"
                        f"  Category: {fallback_category}\n"
                        f"  Classify time: N/A\n"
                        f"  Gemma response: ERROR - {exc}\n"
                        f"  Output path: {destination}"
                    )

            for offset, image_path in enumerate(batch, 1):
                category = "uncategorized"
                raw_response = ""
                classify_seconds = 0.0
                image_start_time = None
                category_already_postprocessed = False
                ollama_error = ollama_errors.get(image_path)

                if image_path in batch_errors:
                    errors_logger.error("Failed to preprocess %s: %s", image_path, batch_errors[image_path])
                    original_size = image_path.stat().st_size
                    payload_size = 0
                else:
                    try:
                        _, original_size, payload_size = batch_cache[image_path]
                        if image_path in classify_results:
                            category, raw_response, image_start_time, classify_seconds = classify_results[image_path]
                            category_already_postprocessed = True
                    except (KeyError, ValueError):
                        errors_logger.error("Failed to preprocess %s: not in cache", image_path)
                        original_size = image_path.stat().st_size
                        payload_size = 0

                total_original_size += original_size
                total_payload_size += payload_size

                if not category_already_postprocessed:
                    category = postprocess_category(category, image_path, categories)
                if category not in output_categories:
                    category = fallback_category

                destination = output / category / image_path.name
                counts[category] += 1
                processed_images += 1

                if dry_run:
                    continue

                if operation == "move":
                    shutil.move(str(image_path), str(destination))
                else:
                    shutil.copy2(image_path, destination)

                if not ollama_error:
                    history_key = history_key_for_image(image_path, source)
                    append_history_entry(effective_history_log_path, history_key, category)
                    history_index[history_key] = category

    except KeyboardInterrupt:
        cancelled = True
        typer.echo("\nCancelando procesamiento... liberando recursos.")
        prep_executor.shutdown(wait=False, cancel_futures=True)
        classify_executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        prep_executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
        classify_executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

    total_seconds = time.perf_counter() - started
    total_images = processed_images

    avg_original_size = (total_original_size / total_images) if total_images else 0.0
    avg_payload_size = (total_payload_size / total_images) if total_images else 0.0

    return Summary(
        counts=counts,
        total_images=total_images,
        skipped_images=skipped_images,
        total_seconds=total_seconds,
        avg_original_size=avg_original_size,
        avg_payload_size=avg_payload_size,
    )
