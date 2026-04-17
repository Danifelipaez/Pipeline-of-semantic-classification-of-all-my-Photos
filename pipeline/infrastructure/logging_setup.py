from __future__ import annotations

import logging
from pathlib import Path


def configure_error_logger(log_file: Path) -> logging.Logger:
    """Create logger writing only error records to target file."""
    logger = logging.getLogger("pipeline_errors")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
