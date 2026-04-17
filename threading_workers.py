"""Threading infrastructure for parallel image processing.

This module implements the infrastructure layer for multi-threaded image
processing, including worker functions and resource management. It separates
domain logic (pure functions) from I/O operations (worker functions).

Per feature-layer-separation skill:
- Domain layer: compute_*, validate_*, format_* (pure functions, no side effects)
- Infrastructure layer: worker_* (handle I/O, threading, cleanup)
- Resource management: ProcessingPipeline context manager (RAII pattern)

Per python-patterns skill:
- Frozen dataclasses prevent accidental mutations in multi-threaded context
- Type hints for static analysis and documentation
- Context managers for resource cleanup guarantee
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ============================================================================
# Domain Layer: Pure Functions (No Side Effects, Thread-Safe)
# ============================================================================


def compute_expected_payload_reduction(
    original_size: int, old_max_side: int = 1024, new_max_side: int = 768
) -> float:
    """Estimate compression ratio when resizing from old to new max_side.

    Args:
        original_size: Original image file size in bytes
        old_max_side: Previous max_side parameter (1024)
        new_max_side: New max_side parameter (768)

    Returns:
        Estimated compression ratio (0.7-0.8 typical for JPEG)

    Pure function: no I/O, no mutations, deterministic.
    """
    # Heuristic: pixel count reduction → file size reduction (squared)
    scale = new_max_side / old_max_side
    reduction_factor = scale ** 2
    return reduction_factor


def validate_classification_confidence(
    confidence: int, threshold: float = 0.85, allow_negative: bool = False
) -> bool:
    """Validate classification confidence against threshold.

    Args:
        confidence: Confidence percentage (0-100)
        threshold: Minimum confidence threshold (0.0-1.0)
        allow_negative: If True, allow confidence < threshold

    Returns:
        True if confidence is acceptable

    Pure function: pure boolean logic.
    """
    if confidence < 0 or confidence > 100:
        return False
    if allow_negative:
        return True
    return confidence / 100.0 >= threshold


def format_classification_output(
    path: Path,
    category: str,
    original_size: int,
    payload_size: int,
    processing_seconds: float,
    raw_response: str,
    error: Exception | None = None,
) -> str:
    """Format classification result for pretty printing.

    Args:
        path: Image path
        category: Classified category
        original_size: Original image size in bytes
        payload_size: Compressed payload size in bytes
        processing_seconds: Processing time
        raw_response: Raw AI response
        error: Optional error message

    Returns:
        Formatted string for printing

    Pure function: string formatting only.
    """
    if error:
        return (
            f"{path.name} | error=ollama | "
            f"Sin respuesta de Ollama, abre tu terminal WSL"
        )

    return (
        f"{path.name} | original_size={original_size} bytes "
        f"| payload_size={payload_size} bytes | category={category} "
        f"| gemma4_response={raw_response} "
        f"| processing_seconds={processing_seconds:.2f} "
        f"| destination={category}/{path.name}"
    )


# ============================================================================
# Data Structures: Immutable Dataclasses (Thread-Safe)
# ============================================================================


@dataclass(frozen=True)
class PreparedImage:
    """Result of image preprocessing: resized and Base64-encoded.

    Frozen=True prevents accidental mutations in multi-threaded context.
    All fields immutable after creation.

    Attributes:
        path: Original image file path
        payload_b64: Base64-encoded JPEG payload
        original_size: Original image file size in bytes
        payload_size: Compressed JPEG size in bytes
        error: Optional exception during preprocessing
    """

    path: Path
    payload_b64: str
    original_size: int
    payload_size: int
    error: Exception | None = None


@dataclass(frozen=True)
class ClassificationResult:
    """Result of AI classification via Ollama.

    Frozen=True prevents accidental mutations in multi-threaded context.
    All fields immutable after creation.

    Attributes:
        path: Original image file path
        original_size: Original image file size in bytes
        payload_size: Compressed JPEG size in bytes
        category: Classified category name
        raw_response: Raw Ollama response text
        seconds: Processing time for classification
        error: Optional exception during classification
    """

    path: Path
    original_size: int
    payload_size: int
    category: str
    raw_response: str
    seconds: float
    error: Exception | None = None  # Use Exception instead of OllamaError to avoid circular import


# ============================================================================
# Resource Management: ProcessingPipeline Context Manager (RAII)
# ============================================================================


class ProcessingPipeline:
    """Context manager for thread pool + bounded queues.

    Guarantees:
    - ThreadPoolExecutor.shutdown(wait=True) called on __exit__
    - All queues drained to force garbage collection
    - No dangling threads or resource leaks

    Per python-patterns skill (RAII pattern): initialize resources in __enter__,
    cleanup in __exit__ (with guarantee even if exception occurs).

    Usage:
        with ProcessingPipeline(num_prep_workers=4) as pipeline:
            # pipeline.executor, pipeline.queues, pipeline.shutdown_event available
            # Automatic cleanup on exit (success or exception)
    """

    def __init__(
        self,
        num_prep_workers: int = 4,
        queue_maxsize: int = 10,
        stats_lock_type: type = threading.Lock,
    ):
        """Initialize pipeline configuration.

        Args:
            num_prep_workers: Number of preprocessing threads (4 recommended)
            queue_maxsize: Maximum items in each queue (10 recommended)
            stats_lock_type: Lock class for stats (threading.Lock or RLock)
        """
        self.num_prep_workers = num_prep_workers
        self.queue_maxsize = queue_maxsize
        self.stats_lock_type = stats_lock_type

        # Initialized in __enter__
        self.executor: ThreadPoolExecutor | None = None
        self.queues: dict[str, queue.Queue] = {}
        self.shutdown_event: threading.Event | None = None
        self.stats_lock: threading.Lock | None = None

    def __enter__(self) -> ProcessingPipeline:
        """Initialize resources."""
        # ThreadPool: 4 prep workers + 1 classify worker
        self.executor = ThreadPoolExecutor(max_workers=self.num_prep_workers + 1)

        # Bounded queues prevent memory bloat
        self.queues = {
            "prep": queue.Queue(maxsize=self.queue_maxsize),
            "inference": queue.Queue(maxsize=self.queue_maxsize),
            "output": queue.Queue(maxsize=self.queue_maxsize),
        }

        # Shutdown signal for graceful termination
        self.shutdown_event = threading.Event()

        # Lock for shared statistics (minimized scope)
        self.stats_lock = self.stats_lock_type()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Cleanup resources (guaranteed even on exception)."""
        # Signal all workers to shutdown
        if self.shutdown_event:
            self.shutdown_event.set()

        # Wait for all submitted tasks to complete
        if self.executor:
            self.executor.shutdown(wait=True)

        # Drain all queues to force garbage collection
        for q in self.queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Return False: re-raise any exception that occurred
        return False


# ============================================================================
# Infrastructure Layer: Worker Functions
# ============================================================================


def producer_worker(
    images: list[Path],
    prep_queue: queue.Queue[Path],
    shutdown_event: threading.Event,
) -> None:
    """Producer: load image paths into prep queue.

    Args:
        images: List of image paths to process
        prep_queue: Queue to enqueue image paths
        shutdown_event: Signal to stop processing

    Side effects:
    - Enqueues Path objects to prep_queue
    - Sends None sentinel to signal end-of-stream
    """
    for img_path in images:
        if shutdown_event.is_set():
            break
        try:
            prep_queue.put(img_path, timeout=1.0)
        except queue.Full:
            # Queue full, wait a bit and retry
            time.sleep(0.1)

    # Send sentinel to signal end
    prep_queue.put(None)


def prep_worker(
    prep_queue: queue.Queue[Path],
    inference_queue: queue.Queue[PreparedImage],
    shutdown_event: threading.Event,
    max_side: int,
    jpeg_quality: int,
    preprocess_fn: Callable,
) -> None:
    """Preprocessing worker: resize and encode images.

    Reads from prep_queue, runs preprocessing, enqueues to inference_queue.

    Args:
        prep_queue: Queue of image paths to process
        inference_queue: Queue to enqueue PreparedImage results
        shutdown_event: Signal to stop processing
        max_side: Maximum image dimension (768)
        jpeg_quality: JPEG quality (80)
        preprocess_fn: Function to preprocess image (closure with params)

    Side effects:
    - Dequeues from prep_queue
    - Enqueues PreparedImage to inference_queue
    - Sends None sentinel when receives None from upstream

    Per python-patterns (EAFP + resource management):
    - Context managers (Image.open) guarantee file cleanup
    - Exceptions captured in dataclass, not thrown
    """
    while not shutdown_event.is_set():
        try:
            img_path = prep_queue.get(timeout=0.5)

            if img_path is None:
                # Sentinel: pass it downstream and exit
                inference_queue.put(None)
                break

            # Perform preprocessing with error handling
            try:
                payload_b64, original_size, payload_size = preprocess_fn(
                    img_path, max_side=max_side, jpeg_quality=jpeg_quality
                )
                result = PreparedImage(
                    path=img_path,
                    payload_b64=payload_b64,
                    original_size=original_size,
                    payload_size=payload_size,
                    error=None,
                )
            except Exception as e:
                # Capture error in dataclass, don't throw
                result = PreparedImage(
                    path=img_path,
                    payload_b64="",
                    original_size=0,
                    payload_size=0,
                    error=e,
                )

            inference_queue.put(result)

        except queue.Empty:
            continue


def classify_worker(
    inference_queue: queue.Queue[PreparedImage],
    output_queue: queue.Queue[ClassificationResult],
    shutdown_event: threading.Event,
    categories: list[str],
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    min_confidence: float,
    fallback_category: str,
    require_confidence_format: bool,
) -> None:
    """Classification worker: send to Ollama, get category.

    Reads from inference_queue, runs Ollama classification,
    enqueues to output_queue.

    Args:
        inference_queue: Queue of PreparedImage to classify
        output_queue: Queue to enqueue ClassificationResult
        shutdown_event: Signal to stop processing
        categories: List of valid categories
        prompt_template: Classification prompt template
        ollama_url: URL to Ollama API
        model: Model name (e.g., "gemma4")
        timeout_seconds: Request timeout
        min_confidence: Minimum confidence threshold
        fallback_category: Category if insufficient confidence
        require_confidence_format: Strict format requirement

    Side effects:
    - Dequeues from inference_queue
    - Makes HTTP POST to Ollama
    - Enqueues ClassificationResult to output_queue
    - Sends None sentinel when receives None from upstream

    Per python-patterns (EAFP + explicit error handling):
    - Exceptions captured in dataclass, not thrown
    """
    # Lazy import to avoid circular import
    from pipeline import OllamaError, classify_with_ollama

    while not shutdown_event.is_set():
        try:
            prep_result = inference_queue.get(timeout=0.5)

            if prep_result is None:
                # Sentinel: pass it downstream and exit
                output_queue.put(None)
                break

            # If preprocessing failed, propagate error
            if prep_result.error:
                result = ClassificationResult(
                    path=prep_result.path,
                    original_size=prep_result.original_size,
                    payload_size=prep_result.payload_size,
                    category="uncategorized",
                    raw_response="",
                    seconds=0.0,
                    error=prep_result.error,
                )
                output_queue.put(result)
                continue

            # Perform classification
            started = time.perf_counter()
            try:
                category, raw_response = classify_with_ollama(
                    prep_result.payload_b64,
                    categories,
                    prompt_template,
                    ollama_url,
                    model,
                    timeout_seconds,
                    min_confidence=min_confidence,
                    fallback_category=fallback_category,
                    require_confidence_format=require_confidence_format,
                )
                elapsed = time.perf_counter() - started

                result = ClassificationResult(
                    path=prep_result.path,
                    original_size=prep_result.original_size,
                    payload_size=prep_result.payload_size,
                    category=category,
                    raw_response=raw_response,
                    seconds=elapsed,
                    error=None,
                )

            except Exception as e:
                elapsed = time.perf_counter() - started
                result = ClassificationResult(
                    path=prep_result.path,
                    original_size=prep_result.original_size,
                    payload_size=prep_result.payload_size,
                    category="uncategorized",
                    raw_response="",
                    seconds=elapsed,
                    error=e,
                )

            output_queue.put(result)

        except queue.Empty:
            continue


def log_collector(
    output_queue: queue.Queue[ClassificationResult],
    pipeline: ProcessingPipeline,
    counts: dict[str, int],
    stats: dict[str, Any],
    operation: str,
    output_path: Path,
    history_log_path: Path | None,
    postprocess_fn: Callable,
    append_history_fn: Callable,
    errors_logger: Any = None,
    categories: list[str] | None = None,
) -> None:
    """Output logger: consume results, write files, update history.

    Runs in main thread (sequential I/O operations).
    Responsible for:
    - Printing formatted output
    - Copying/moving files to destination
    - Updating history.log
    - Updating statistics counters (with lock)

    Args:
        output_queue: Queue of ClassificationResult to log
        pipeline: ProcessingPipeline context with stats_lock
        counts: Category counters dict
        stats: Statistics dict (total_original_size, total_payload_size)
        operation: "copy", "move" or "none"
        output_path: Destination folder path
        history_log_path: Path to history.log (None if dry_run)
        postprocess_fn: Function to postprocess category
        append_history_fn: Function to append entry to history.log
        errors_logger: Optional logger for errors
        categories: List of categories for postprocessing

    Side effects:
    - Prints to stdout
    - Writes to destination folders (copy/move)
    - Appends to history.log
    - Updates counters with lock

    Per python-patterns (explicit error handling):
    - Wraps network/file I/O, doesn't throw to caller
    """
    import shutil  # Import here to avoid circular dependency

    while not pipeline.shutdown_event.is_set():
        try:
            result = output_queue.get(timeout=1.0)

            if result is None:
                # Sentinel: exit
                break

            # Postprocess category (heuristic by filename)
            category = postprocess_fn(result.category, result.path, categories or [])

            # Update statistics with lock
            with pipeline.stats_lock:
                counts[category] += 1
                stats["total_original_size"] += result.original_size
                stats["total_payload_size"] += result.payload_size

            # Print formatted output
            formatted = format_classification_output(
                result.path,
                category,
                result.original_size,
                result.payload_size,
                result.seconds,
                result.raw_response,
                result.error,
            )
            print(formatted)

            # File operations (sequential, in main thread)
            if not result.error and operation != "none":
                destination = output_path / category / result.path.name

                try:
                    if operation == "move":
                        shutil.move(str(result.path), str(destination))
                    else:
                        shutil.copy2(result.path, destination)

                    # Update history only after successful file operation
                    if history_log_path:
                        append_history_fn(history_log_path, result.path.name, category)

                except Exception as e:
                    msg = f"{result.path.name} | error=file_operation | {e}"
                    print(msg)
                    if errors_logger:
                        errors_logger.error(msg)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in log_collector: {e}")
            if errors_logger:
                errors_logger.error(f"Error in log_collector: {e}")
            continue
