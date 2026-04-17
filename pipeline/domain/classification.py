from __future__ import annotations

import re
from pathlib import Path


def extract_category(
    response_text: str,
    categories: list[str],
    *,
    min_confidence: float = 0.85,
    fallback_category: str = "uncategorized",
    require_confidence_format: bool = True,
) -> str:
    """Extract category and confidence from model output.

    If confidence is below threshold or format is invalid, fallback category is returned.
    """
    normalized_categories = {re.sub(r"\s+", " ", c.lower().strip()): c for c in categories}
    normalized_fallback = fallback_category.strip() or "uncategorized"

    match = re.search(r"([a-zA-Z0-9 _-]+)\s*\|\s*<?(\d{1,3})>?", response_text.strip())
    if match:
        cat_raw, conf_raw = match.groups()
        try:
            confidence = int(conf_raw) / 100.0
        except Exception:
            confidence = 0.0

        normalized = cat_raw.lower().strip()
        if normalized in normalized_categories and confidence >= min_confidence:
            return normalized_categories[normalized]
        return normalized_fallback

    if require_confidence_format:
        return normalized_fallback

    normalized_response = response_text.lower().strip()
    normalized_response = re.sub(r"[^a-z0-9\s]", " ", normalized_response)
    normalized_response = re.sub(r"\s+", " ", normalized_response).strip()

    if normalized_response in normalized_categories:
        return normalized_categories[normalized_response]

    for normalized, original in normalized_categories.items():
        pattern = rf"\b{re.escape(normalized)}\b"
        if re.search(pattern, normalized_response):
            return original

    first_word = normalized_response.split(" ")[0] if normalized_response else ""
    for normalized, original in normalized_categories.items():
        if normalized.split(" ")[0] == first_word and first_word:
            return original

    return normalized_fallback


def build_effective_prompt(prompt_template: str, categories: list[str], fallback_category: str) -> str:
    """Build prompt with strict output contract category|confidence."""
    category_list = ", ".join(categories)
    base_prompt = prompt_template.format(categories=category_list).strip()
    contract_instructions = (
        "\n\nOutput contract (strict): reply with exactly one line in this format: "
        "<category>|<confidence_0_to_100>. "
        f"If confidence is low, use {fallback_category}|<confidence>. "
        "Do not add explanations or extra text."
    )
    return f"{base_prompt}{contract_instructions}"


def postprocess_category(category: str, image_path: Path, categories: list[str]) -> str:
    """Apply filename-based heuristic when category is uncategorized."""
    if category != "uncategorized":
        return category

    filename = image_path.stem.lower()
    for cat in categories:
        if cat.lower() in filename:
            return cat
    return category
