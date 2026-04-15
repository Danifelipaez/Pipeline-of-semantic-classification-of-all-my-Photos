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
- classification behavior (`classification.min_confidence`, `classification.fallback_category`, `classification.require_confidence_format`)

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

## Flujo del pipeline

1. **Carga de configuración**
   - Se lee `config.yaml` para obtener categorías, rutas, parámetros de preprocesado y configuración de Ollama.

2. **Listado de imágenes**
   - Se buscan todas las imágenes soportadas en la carpeta de origen (recursivo).

3. **Preprocesamiento**
   - Cada imagen se abre y convierte a RGB si es necesario.
   - Se redimensiona para que el lado más largo no supere el valor configurado (`max_side`).
   - Se guarda temporalmente como JPEG con la calidad indicada.
   - Se codifica la imagen resultante en base64 para enviarla por API.

4. **Clasificación con Gemma4 (Ollama)**
   - Se construye un prompt con las categorías y se envía la imagen codificada a la API de Ollama.
   - El pipeline fuerza un contrato de salida estricto: `categoria|confianza`.
   - Se recibe la respuesta del modelo y se parsea categoría + confianza.
   - Si la confianza es menor al umbral (por defecto 85%), se marca como `uncategorized`.
   - **Si Ollama no responde o hay timeout/conexión rechazada:** se muestra en terminal "Sin respuesta de Ollama, abre tu terminal WSL" y la foto NO se registra en history.log (permitiendo reintentos posteriores).

5. **Postproceso**
   - Si la imagen fue clasificada como `uncategorized`, se intenta reasignar una categoría si el nombre del archivo contiene alguna palabra clave de las categorías.

6. **Organización de archivos**
   - Según la categoría final, la imagen se copia o mueve a la carpeta correspondiente en el destino.
   - Si ocurre un error, se registra en `errors.log`.

7. **Resumen**
   - Al finalizar, se imprime un resumen con el número de imágenes por categoría, tiempo total y tamaños promedio.
