from __future__ import annotations

from pathlib import Path


def load_history_index(history_log_path: Path) -> dict[str, str]:
    """Load processed image history as case-insensitive key -> category index."""
    history_index: dict[str, str] = {}
    if not history_log_path.exists():
        return history_index

    with history_log_path.open("r", encoding="utf-8") as history_file:
        for raw_line in history_file:
            if "|" not in raw_line:
                continue
            filename, category = raw_line.split("|", maxsplit=1)
            normalized_name = filename.strip()
            normalized_category = category.strip()
            if normalized_name:
                history_index[normalized_name.casefold()] = normalized_category
    return history_index


def append_history_entry(history_log_path: Path, filename: str, category: str) -> None:
    """Append one processed image entry to history log."""
    history_log_path.parent.mkdir(parents=True, exist_ok=True)
    with history_log_path.open("a", encoding="utf-8") as history_file:
        history_file.write(f"{filename} | {category}\n")


def history_key_for_image(image_path: Path, source_root: Path) -> str:
    """Build stable case-insensitive source-relative history key."""
    try:
        relative = image_path.relative_to(source_root)
        normalized = str(relative).replace("\\", "/")
    except ValueError:
        normalized = image_path.name
    return normalized.casefold()
