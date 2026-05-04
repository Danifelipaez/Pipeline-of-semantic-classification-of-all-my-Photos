from __future__ import annotations

import requests
from pathlib import Path
from typing import Any, Dict


class OllamaClient:
    def __init__(self, url: str, model: str, timeout_seconds: int):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, image_b64: str | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if image_b64 is not None:
            payload["images"] = [image_b64]
        resp = requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()
