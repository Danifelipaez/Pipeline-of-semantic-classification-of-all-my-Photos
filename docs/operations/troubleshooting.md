# Operación y Troubleshooting

Guía práctica para fallos comunes del pipeline. Este documento reemplaza la información dispersa en README y los resúmenes antiguos.

## Ollama no responde

Síntomas:

- `ConnectionError`
- `Timeout`
- respuesta vacía o error HTTP

Acciones:

1. Verifica que `ollama serve` esté activo.
2. Confirma que `http://localhost:11434/api/tags` responde.
3. Aumenta `ollama.timeout_seconds` si las imágenes son grandes o el hardware es limitado.
4. Reduce `preprocessing.max_side` si necesitas menos payload.

## Muchas imágenes van a `uncategorized`

Causas típicas:

- el modelo devuelve una categoría fuera de la lista válida;
- la confianza cae por debajo del umbral;
- la respuesta no respeta el contrato.

Acciones:

1. Revisa `classification.min_confidence`.
2. Mantén `classification.require_confidence_format = true`.
3. Ajusta el prompt para reforzar el contrato `categoria|confianza`.
4. Si hace falta, amplía o corrige la lista de categorías.

## Reprocesado inesperado

Causas típicas:

- historial viejo con nombres no normalizados;
- archivos movidos entre carpetas fuente;
- diferencias de mayúsculas/minúsculas.

Acciones:

1. Verifica `history.log`.
2. Evita duplicar nombres base en carpetas distintas si el historial aún contiene entradas antiguas.
3. Usa rutas relativas estables cuando sea posible.

## La memoria crece demasiado

Acciones:

1. Reduce `prep_batch_size` si la fuente es muy grande.
2. Limita `classification_workers` cuando el hardware sea modesto.
3. Mantén la ejecución por lotes y evita cargar toda la colección en memoria.

## Errores de configuración

Mensajes relevantes:

- `operation must be 'copy' or 'move'`
- `classification.min_confidence must be between 0 and 1`
- `Config must define at least one category`

Acciones:

1. Corrige `config.yaml`.
2. Revisa el documento canónico de configuración.
3. Ejecuta una corrida pequeña antes de procesar todo el catálogo.
