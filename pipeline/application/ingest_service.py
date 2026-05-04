from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..domain.models import PhotoRecord
from ..infrastructure.persistence import Persistence


def ingest_and_index(
    *,
    source: Path,
    max_side: int,
    jpeg_quality: int,
    prompt_template: str,
    ollama_url: str,
    model: str,
    timeout_seconds: int,
    fallback_text: str,
    require_json: bool,
    dry_run: bool,
    description_log_path: Path,
    db_path: Path,
):
    # call existing process_images implementation
    # we import here to avoid circular imports
    from pipeline import process_images
    summary = process_images(
        source=source,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        prompt_template=prompt_template,
        ollama_url=ollama_url,
        model=model,
        timeout_seconds=timeout_seconds,
        fallback_text=fallback_text,
        require_json=require_json,
        dry_run=dry_run,
        description_log_path=description_log_path,
    )

    # index descriptions.jsonl into SQLite
    persistence = Persistence(db_path)
    records: List[PhotoRecord] = []
    if description_log_path.exists():
        with description_log_path.open("r", encoding="utf-8") as f:
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
                persistence.upsert_photo(rec)

    return summary
