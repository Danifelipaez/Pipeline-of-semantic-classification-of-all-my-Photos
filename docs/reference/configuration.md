# Referencia de Configuración

Este es el documento canónico para el contrato de `config.yaml` y sus valores por defecto. Si algún valor cambia en el código, esta referencia debe actualizarse en primer lugar.

## Propósito

Centralizar los defaults, rangos válidos y reglas de comportamiento de la clasificación para evitar drift entre README, CLAUDE y documentos de optimización.

## Valores por defecto

```yaml
source: ./photos
output: ./sorted
operation: copy
categories:
  - nature
  - birds
  - street photography
  - family
  - portraits
  - architecture
  - food
  - events
ollama:
  url: http://localhost:11434
  model: gemma4
  timeout_seconds: 120
preprocessing:
  max_side: 1024
  jpeg_quality: 80
classification:
  min_confidence: 0.85
  fallback_category: uncategorized
  require_confidence_format: true
  workers_mode: auto
  num_workers: 4
  max_workers_cap: 6
```

## Reglas de validación

- `categories` no puede quedar vacía.
- `operation` solo acepta `copy` o `move`.
- `classification.min_confidence` debe estar entre 0 y 1.
- `classification.fallback_category` no puede estar vacío.
- `classification.workers_mode` solo acepta `auto` o `manual`.
- `classification.num_workers` y `classification.max_workers_cap` deben estar entre 1 y 64.

## Contrato de clasificación

La salida esperada de Ollama es `categoria|confianza`, donde la confianza está entre 0 y 100.

Reglas relevantes:

- Si la respuesta no respeta el formato y `require_confidence_format` es `true`, se usa `fallback_category`.
- Si la categoría no pertenece a la lista válida, se usa `fallback_category`.
- Si la confianza es menor al umbral configurado, se usa `fallback_category`.
- El valor por defecto de `fallback_category` es `uncategorized`.

## Ejemplo recomendado

```yaml
classification:
  min_confidence: 0.85
  fallback_category: uncategorized
  require_confidence_format: true
  workers_mode: auto
  num_workers: 4
  max_workers_cap: 6
```

## Notas operativas

- El preprocesado reduce tamaño antes de enviar la imagen a Ollama.
- La clasificación puede usar paralelismo configurable mediante `classification_workers` y `resolve_classification_workers()`.
- Si necesitas ajustar el comportamiento de IA, modifica primero este contrato y luego actualiza README y CLAUDE.
