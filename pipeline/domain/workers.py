from __future__ import annotations

import os
from typing import Any


def resolve_classification_workers(classification_config: dict[str, Any]) -> int:
    """Resolve worker count based on config mode and host capacity."""
    mode = str(classification_config.get("workers_mode", "auto")).strip().lower()
    configured_workers = int(classification_config.get("num_workers", 4))
    workers_cap = int(classification_config.get("max_workers_cap", 6))

    if mode == "manual":
        return max(1, min(configured_workers, workers_cap))

    cpu_count = os.cpu_count() or 2
    if cpu_count <= 2:
        auto_workers = 1
    elif cpu_count <= 4:
        auto_workers = 1
    elif cpu_count <= 6:
        auto_workers = 2
    elif cpu_count <= 10:
        auto_workers = 2
    else:
        auto_workers = 3

    try:
        import psutil  # type: ignore

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 4:
            auto_workers = 1
        elif available_gb < 8:
            auto_workers = min(auto_workers, 2)
    except Exception:
        pass

    return max(1, min(auto_workers, workers_cap))
