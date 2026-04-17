# Optimization Summary

Documento canónico de historial de optimización y decisiones de performance.

## Resumen de cambios

- Preprocesado en paralelo con 4 workers.
- Reducción de `max_side` y `jpeg_quality` para bajar payload.
- Clasificación con workers configurables, no hardcodeada.
- Contrato estricto de salida `categoria|confianza`.
- Mejor visibilidad de progreso durante la ejecución.
- Tests ampliados para concurrencia, limpieza y comportamiento de configuración.

## Decisiones que se mantienen

- El cuello de botella principal sigue siendo Ollama.
- El preprocesado debe seguir siendo memoria-safe y sin mutar archivos originales.
- La documentación de configuración vive en docs/reference/configuration.md.
- La documentación operativa vive en docs/operations/troubleshooting.md.

## Estado actual

Este archivo es el punto de referencia para leer el historial de optimización. Los documentos antiguos en la raíz deben considerarse compatibilidad o archivo histórico.
