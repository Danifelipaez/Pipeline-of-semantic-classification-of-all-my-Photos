from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from ..domain.models import FolderSummary, PhotoRecord
from ..infrastructure.persistence import Persistence


class IndexService:
    def __init__(self, db_path: Path):
        self.persistence = Persistence(db_path)

    def index_photos(self, records: Iterable[PhotoRecord]) -> None:
        for rec in records:
            self.persistence.upsert_photo(rec)

    def index_folder_summary(self, folder: FolderSummary) -> None:
        self.persistence.upsert_folder(folder)
