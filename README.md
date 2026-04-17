# Pipeline de Clasificación Semántica de Fotos

Pipeline local para organizar fotos por categoría semántica usando Ollama y un modelo de visión.

## Qué hace

- Clasifica imágenes `.jpg`, `.jpeg`, `.png` y `.webp`.
- Preprocesa cada imagen en memoria sin modificar el archivo original durante la inferencia.
- Copia o mueve la imagen al directorio de salida según la categoría final.
- Registra errores en `errors.log` y evita reprocesar imágenes ya vistas mediante `history.log`.

## Inicio rápido

1. Crea y activa el entorno Python.
2. Instala dependencias con `pip install -r requirements.txt`.
3. Inicia Ollama en `http://localhost:11434`.
4. Ejecuta `python main.py --source ./photos --output ./sorted`.

## Requisitos

- Python 3.11+
- Ollama local en ejecución
- Modelo de visión descargado en Ollama

## Configuración principal

El contrato canónico vive en [docs/reference/configuration.md](docs/reference/configuration.md).

Valores relevantes por defecto:

- `operation`: `copy`
- `ollama.model`: `gemma4`
- `preprocessing.max_side`: `1024`
- `preprocessing.jpeg_quality`: `85`
- `classification.min_confidence`: `0.85`
- `classification.fallback_category`: `uncategorized`
- `classification.require_confidence_format`: `true`

## Documentación técnica

- [Arquitectura del pipeline](docs/reference/architecture.md)
- [Configuración canónica](docs/reference/configuration.md)
- [Operación y troubleshooting](docs/operations/troubleshooting.md)
- [Historial de optimización](docs/archive/optimization-summary.md)

## Ejecución

```bash
python main.py --source ./photos --output ./sorted
```

Modo de prueba:

```bash
python main.py --source ./photos --output ./sorted --dry-run
```

## Notas

- La clasificación puede usar paralelismo configurable.
- La confianza baja o una respuesta inválida caen en `uncategorized`.
- La heurística de nombre de archivo solo actúa como postproceso.
