from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from ..domain.models import PhotoRecord
from ..infrastructure.ollama_client import OllamaClient
from ..infrastructure.persistence import Persistence


class SearchService:
    def __init__(self, db_path: Path, ollama: OllamaClient):
        self.persistence = Persistence(db_path)
        self.ollama = ollama

    def text_search(self, query: str, limit: int = 50) -> List[Tuple[int, str, str]]:
        return self.persistence.search_text(query, limit=limit)

    def semantic_rank(self, query: str, candidates: List[Tuple[int, str, str]]) -> List[Tuple[int, str, float]]:
        # candidates: list of (id, relative_path, description)
        # Build prompt for Ollama to score candidates
        prompt_lines = [f"Query: {query}", "Candidates:"]
        for i, (_, path, desc) in enumerate(candidates, start=1):
            prompt_lines.append(f"{i}. {path}: {desc}")
        prompt_lines.append(
            "Return JSON: {\"scores\": [float,...]} where scores are 0-100 relevance."
        )
        prompt = "\n".join(prompt_lines)
        resp = self.ollama.generate(prompt)
        raw = resp.get("response", "")
        try:
            parsed = raw if isinstance(raw, dict) else __import__("json").loads(raw)
            scores = parsed.get("scores", [])
        except Exception:
            scores = []

        ranked = []
        for (rowid, path, _), score in zip(candidates, scores):
            ranked.append((rowid, path, float(score)))
        # fallback: if no scores returned, give equal score 0
        if not ranked:
            ranked = [(rowid, path, 0.0) for (rowid, path, _) in candidates]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked
