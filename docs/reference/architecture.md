# Arquitectura del Pipeline

Este documento describe la estructura lógica del sistema y cómo se relacionan sus capas. La intención es que cada responsabilidad tenga un solo lugar de cambio.

## Capas

### CLI

- `main.py` maneja argumentos, entrada de usuario y ejecución del flujo.
- No debe contener reglas de dominio ni decisiones de clasificación.

### Aplicación

- `process_images()` coordina el escaneo, preprocesado, clasificación, organización de archivos e historial.
- Decide el orden de ejecución y conecta dependencias.

### Dominio

Funciones puras o casi puras:

- `load_config()` valida y normaliza la configuración.
- `extract_category()` interpreta la respuesta del modelo.
- `postprocess_category()` aplica la heurística del nombre de archivo.
- `resolve_classification_workers()` decide el número efectivo de workers.

### Infraestructura

- `preprocess_image_for_inference()` usa PIL para preparar la imagen.
- `classify_with_ollama()` llama a la API local de Ollama.
- La capa de archivos usa `shutil`, `Path` y logs persistentes.

## Flujo de ejecución

1. Cargar configuración.
2. Preparar carpetas de salida.
3. Cargar `history.log` para evitar reprocesar.
4. Escanear imágenes soportadas.
5. Preprocesar en paralelo.
6. Clasificar con workers configurables.
7. Aplicar postproceso y copiar o mover el archivo.
8. Registrar éxito o error.
9. Emitir resumen final.

## Estrategia de concurrencia

- El preprocesado usa un pool fijo de 4 threads.
- La clasificación usa un pool configurado según `classification_workers`.
- La persistencia de historial sigue siendo secuencial para evitar corrupción.
- Los contadores y salidas concurrentes deben protegerse con lock cuando haya escritura compartida.

## Contratos importantes

- Nunca modificar la imagen original durante inferencia.
- No registrar en history un error de Ollama que deba poder reintentarse.
- Mantener `fallback_category` como destino seguro para respuestas inválidas o de baja confianza.
- Priorizar separación de responsabilidades: la lógica de negocio no debe mezclarse con el I/O.
