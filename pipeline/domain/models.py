from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PhotoRecord:
    relative_path: str
    filename: str
    source_path: str
    output_path: str
    description: str
    tags: List[str]
    entities: Dict[str, List[str]]
    scene_attributes: Dict[str, Any]
    raw_response: str


@dataclass
class FolderSummary:
    folder_path: str
    summary_text: str
    tag_histogram: Dict[str, int]
    entity_summary: Dict[str, int]
    photo_count: int
    last_updated: float
