# Pipeline-of-semantic-classification-of-all-my-Photos

Local photo organization pipeline using Gemma4 through Ollama.

## Features

- Generates semantic descriptions + tags for `.jpg`, `.jpeg`, `.png`, and `.webp` images
- Uses Ollama local API (`http://localhost:11434`) with `gemma4`
- Writes a `descriptions.jsonl` log for chatbot-friendly grouping
- Logs failed requests to `errors.log`
- Keeps originals untouched during inference (in-memory preprocessing only)
- Stores `descriptions.jsonl`, `errors.log`, and `index.db` in `source`
- Supports `--dry-run`
- Prints summary with totals, time, and average payload vs original size

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
3. Pull Gemma4:
   ```bash
   ollama pull gemma4
   ```

## Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:

- `source`
- preprocessing (`max_side`, `jpeg_quality`)
- Ollama URL/model/timeout
- `description_prompt` (prompt for semantic descriptions)
- `description.fallback_text` and `description.require_json`
- `description_output` (file name for the JSONL log)

## Nuevas funciones: indexación y búsqueda

- Indexado local: el pipeline ahora genera un índice SQLite embebido (`index.db`) dentro de la carpeta `source`.
- Resúmenes por foto enriquecidos: cada entrada en `descriptions.jsonl` contiene ahora campos adicionales (estructurados) para facilitar búsquedas:
   - `entities` (personas, objetos, lugares, actividades)
   - `scene_attributes` (lighting, weather, colors, time_of_day)
   - `searchable_description` (frase corta optimizada para recuperación)
- Búsqueda semántica: `main.py` ofrece un comando `search` que combina búsqueda textual (SQLite FTS5) con un re-ranking semántico vía Gemma4.

## Nuevos comandos CLI

- Ingest / procesado (mantiene la interfaz anterior):

```bash
python main.py --source ./photos
```

- Rebuild index (reconstruye `index.db` a partir de `descriptions.jsonl`):

```bash
python main.py rebuild-index
```

- Search (texto; opción semántica con Gemma4):

```bash
python main.py search "retratos de mujeres al atardecer con el mar de fondo" --semantic
```

La búsqueda sin `--semantic` usa FTS5 para rapidez; con `--semantic` se añade un re-ranking por similitud usando Gemma4 a través de Ollama.

## Prompt y contrato para Gemma4

El prompt enviado a Gemma4 solicita ahora una salida JSON estricta y estructurada (una sola línea) que incluye `description`, `tags`, `entities`, `scene_attributes` y `searchable_description`. Esto mejora la precisión de las búsquedas en lenguaje natural y permite realizar re-rankings semánticos.

Ejemplo (interno): el prompt exige "NO añadir texto adicional, sólo un JSON válido en una línea" para preservar la compatibilidad con el parser del pipeline.

## Run

```bash
python main.py --source ./photos
```

Dry run:

```bash
python main.py --source ./photos --dry-run
```

The pipeline preprocesses each image in memory only:

- converts temporary inference copy to RGB
- resizes longest side to max 1024 (default) with `Image.LANCZOS`
- encodes temporary copy as JPEG (quality 85 by default)
- sends base64 payload to Ollama

Original files are never copied or moved by the pipeline.

## Flujo del pipeline

1. **Carga de configuración**
   - Se lee `config.yaml` para obtener rutas, parámetros de preprocesado, prompt de descripción y configuración de Ollama.

2. **Listado de imágenes**
   - Se buscan todas las imágenes soportadas en la carpeta de origen (recursivo).

3. **Preprocesamiento**
   - Cada imagen se abre y convierte a RGB si es necesario.
   - Se redimensiona para que el lado más largo no supere el valor configurado (`max_side`).
   - Se guarda temporalmente como JPEG con la calidad indicada.
   - Se codifica la imagen resultante en base64 para enviarla por API.

4. **Descripción semántica con Gemma4 (Ollama)**
   - Se construye un prompt y se envía la imagen codificada a la API de Ollama.
   - El pipeline fuerza un contrato de salida JSON con `description` y `tags`.
   - Se guarda la respuesta en `descriptions.jsonl` para uso por un chatbot.
   - **Si Ollama no responde o hay timeout/conexión rechazada:** se muestra en terminal "Sin respuesta de Ollama, abre tu terminal WSL" y la foto NO se registra en `descriptions.jsonl` (permitiendo reintentos posteriores).

5. **Organización de archivos**
   - La imagen permanece en su ubicación original (solo se generan descripciones e índice).
   - Si ocurre un error, se registra en `errors.log` dentro de `source`.

6. **Resumen**
   - Al finalizar, se imprime un resumen con totales, tiempo y tamaños promedio.
