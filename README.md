# Pipeline-of-semantic-classification-of-all-my-Photos

Local photo organization pipeline using Gemma 3 through Ollama.

## Features

- Classifies `.jpg`, `.jpeg`, `.png`, and `.webp` images from a source folder
- Uses Ollama local API (`http://localhost:11434`) with `gemma3`
- Sorts photos into category folders (or `uncategorized`)
- Logs failed requests to `errors.log`
- Keeps originals untouched during inference (in-memory preprocessing only)
- Supports copy or move modes
- Supports `--dry-run`
- Prints summary with category counts, total time, and average payload vs original size

## Requirements

- Python 3.11+
- WSL (Ubuntu recommended)
- Ollama installed in WSL

## WSL + Ollama setup

1. Install Ollama in WSL:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Start Ollama:
   ```bash
   ollama serve
   ```
3. Pull Gemma 3:
   ```bash
   ollama pull gemma3
   ```

## Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:

- `categories`
- `source`
- `output`
- `operation` (`copy` or `move`)
- preprocessing (`max_side`, `jpeg_quality`)
- Ollama URL/model/timeout

## Run

```bash
python main.py --source ./photos --output ./sorted
```

Dry run:

```bash
python main.py --source ./photos --output ./sorted --dry-run
```

The pipeline preprocesses each image in memory only:

- converts temporary inference copy to RGB
- resizes longest side to max 1024 (default) with `Image.LANCZOS`
- encodes temporary copy as JPEG (quality 85 by default)
- sends base64 payload to Ollama

Original files are only copied or moved after classification.
