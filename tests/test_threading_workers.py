"""Tests for threading infrastructure and memory-safe optimizations."""

import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from threading_workers import (
    ClassificationResult,
    PreparedImage,
    ProcessingPipeline,
)


def test_prepared_image_is_frozen() -> None:
    """PreparedImage dataclass should be frozen (immutable)."""
    img = PreparedImage(
        path=Path("test.jpg"),
        payload_b64="abc123",
        original_size=1000,
        payload_size=500,
    )

    # Trying to mutate should raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        img.original_size = 2000  # type: ignore


def test_prepared_image_default_error_is_none() -> None:
    """PreparedImage error field should default to None."""
    img = PreparedImage(
        path=Path("test.jpg"),
        payload_b64="abc",
        original_size=100,
        payload_size=50,
    )
    assert img.error is None


def test_classification_result_is_frozen() -> None:
    """ClassificationResult dataclass should be frozen (immutable)."""
    result = ClassificationResult(
        path=Path("test.jpg"),
        original_size=1000,
        payload_size=500,
        category="nature",
        raw_response="nature|92",
        seconds=5.5,
    )

    # Trying to mutate should raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        result.category = "birds"  # type: ignore


def test_classification_result_default_error_is_none() -> None:
    """ClassificationResult error field should default to None."""
    result = ClassificationResult(
        path=Path("test.jpg"),
        original_size=1000,
        payload_size=500,
        category="nature",
        raw_response="nature|92",
        seconds=5.5,
    )
    assert result.error is None


def test_processing_pipeline_context_manager() -> None:
    """ProcessingPipeline context manager should initialize and cleanup resources."""
    pipeline = ProcessingPipeline(num_prep_workers=4, queue_maxsize=10)

    # Before entering context, resources should be None
    assert pipeline.executor is None
    assert not pipeline.queues
    assert pipeline.shutdown_event is None
    assert pipeline.stats_lock is None

    with pipeline:
        # Inside context, resources should be initialized
        assert pipeline.executor is not None
        assert len(pipeline.queues) == 3  # prep, inference, output
        assert pipeline.shutdown_event is not None
        assert pipeline.stats_lock is not None
        assert not pipeline.shutdown_event.is_set()

        # executor should be active
        assert not pipeline.executor._shutdown

        # Queues should be bounded
        assert pipeline.queues["prep"].maxsize == 10
        assert pipeline.queues["inference"].maxsize == 10
        assert pipeline.queues["output"].maxsize == 10

    # After exiting context, executor should be shut down
    assert pipeline.executor._shutdown


def test_processing_pipeline_shutdown_event_signals_workers() -> None:
    """Shutdown event should be accessible and settable by workers."""
    with ProcessingPipeline() as pipeline:
        assert not pipeline.shutdown_event.is_set()

        # Simulate worker checking shutdown
        pipeline.shutdown_event.set()
        assert pipeline.shutdown_event.is_set()


def test_processing_pipeline_stats_lock_is_thread_safe() -> None:
    """Stats lock should be a valid threading lock with acquire/release."""
    with ProcessingPipeline() as pipeline:
        # Should have acquire and release methods (duck typing for Lock)
        assert hasattr(pipeline.stats_lock, "acquire")
        assert hasattr(pipeline.stats_lock, "release")

        # Should be acquirable
        acquired = pipeline.stats_lock.acquire(timeout=1.0)
        assert acquired
        pipeline.stats_lock.release()


def test_processing_pipeline_queues_are_bounded() -> None:
    """Queues should have maxsize set to prevent memory bloat."""
    maxsize = 5
    with ProcessingPipeline(queue_maxsize=maxsize) as pipeline:
        for q in pipeline.queues.values():
            assert q.maxsize == maxsize


def test_processing_pipeline_exception_in_context_still_cleans_up() -> None:
    """Context manager should cleanup even if exception occurs inside."""
    pipeline = ProcessingPipeline(num_prep_workers=2)

    try:
        with pipeline:
            assert pipeline.executor is not None
            # Simulate exception inside context
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Executor should still be shut down
    assert pipeline.executor._shutdown


def test_processing_pipeline_queues_drained_after_exit() -> None:
    """Context manager should drain queues after exit to force garbage collection."""
    with ProcessingPipeline() as pipeline:
        # Put some items in queues
        pipeline.queues["prep"].put(Path("test1.jpg"))
        pipeline.queues["prep"].put(Path("test2.jpg"))
        assert pipeline.queues["prep"].qsize() == 2

    # After exit, queues should be empty (drained)
    assert pipeline.queues["prep"].empty()


def test_processing_pipeline_multiple_contexts_independent() -> None:
    """Multiple ProcessingPipeline contexts should be independent."""
    with ProcessingPipeline(num_prep_workers=3) as p1:
        with ProcessingPipeline(num_prep_workers=5) as p2:
            # Both should have independent executors
            assert p1.executor is not p2.executor
            assert p1.queues is not p2.queues
            assert p1.shutdown_event is not p2.shutdown_event
