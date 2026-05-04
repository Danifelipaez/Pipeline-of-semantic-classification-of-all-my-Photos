from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from ..domain.models import PhotoRecord, FolderSummary


class Persistence:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY,
                relative_path TEXT UNIQUE,
                folder_path TEXT,
                description TEXT,
                tags TEXT,
                entities TEXT,
                scene_attributes TEXT,
                raw_response TEXT,
                indexed_at REAL
            );
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY,
                folder_path TEXT UNIQUE,
                summary_text TEXT,
                tag_histogram TEXT,
                entity_summary TEXT,
                photo_count INTEGER,
                last_updated REAL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
                description, tags, entities, content='photos', content_rowid='id'
            );
            """
        )
        self.conn.commit()

    def upsert_photo(self, record: PhotoRecord) -> None:
        cur = self.conn.cursor()
        now = time.time()
        cur.execute(
            """
            INSERT INTO photos (relative_path, folder_path, description, tags, entities, scene_attributes, raw_response, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path) DO UPDATE SET
              description=excluded.description,
              tags=excluded.tags,
              entities=excluded.entities,
              scene_attributes=excluded.scene_attributes,
              raw_response=excluded.raw_response,
              indexed_at=excluded.indexed_at
            """,
            (
                record.relative_path,
                str(Path(record.relative_path).parent),
                record.description,
                json.dumps(record.tags, ensure_ascii=False),
                json.dumps(record.entities, ensure_ascii=False),
                json.dumps(record.scene_attributes, ensure_ascii=False),
                record.raw_response,
                now,
            ),
        )
        # update FTS
        cur.execute("DELETE FROM photos_fts WHERE rowid = (SELECT id FROM photos WHERE relative_path=?)", (record.relative_path,))
        cur.execute("INSERT INTO photos_fts(rowid, description, tags, entities) SELECT id, description, tags, entities FROM photos WHERE relative_path=?", (record.relative_path,))
        self.conn.commit()

    def upsert_folder(self, summary: FolderSummary) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO folders (folder_path, summary_text, tag_histogram, entity_summary, photo_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(folder_path) DO UPDATE SET
              summary_text=excluded.summary_text,
              tag_histogram=excluded.tag_histogram,
              entity_summary=excluded.entity_summary,
              photo_count=excluded.photo_count,
              last_updated=excluded.last_updated
            """,
            (
                summary.folder_path,
                summary.summary_text,
                json.dumps(summary.tag_histogram, ensure_ascii=False),
                json.dumps(summary.entity_summary, ensure_ascii=False),
                summary.photo_count,
                summary.last_updated,
            ),
        )
        self.conn.commit()

    def search_text(self, query: str, limit: int = 50) -> List[Tuple[int, str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT photos.id, photos.relative_path, photos.description FROM photos_fts JOIN photos ON photos_fts.rowid=photos.id WHERE photos_fts MATCH ? LIMIT ?",
            (query, limit),
        )
        return cur.fetchall()

    def list_folder_candidates(self, query: str, limit: int = 50) -> List[Tuple[str, float]]:
        # simple text match against summary_text
        cur = self.conn.cursor()
        cur.execute("SELECT folder_path, last_updated FROM folders WHERE summary_text LIKE ? LIMIT ?", (f"%{query}%", limit))
        return cur.fetchall()
