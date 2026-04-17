# CLAUDE.md

Índice técnico para agentes y mantenedores. Este archivo no debe duplicar la documentación canónica; solo debe orientar hacia ella.

## Fuente de verdad

- [Configuración](docs/reference/configuration.md)
- [Arquitectura](docs/reference/architecture.md)
- [Operación y troubleshooting](docs/operations/troubleshooting.md)
- [Historial de optimización](docs/archive/optimization-summary.md)

## Reglas que no deben romperse

- La salida de Ollama debe seguir el contrato `categoria|confianza`.
- `classification.require_confidence_format` debe permanecer en `true` salvo cambio explícito de diseño.
- `uncategorized` sigue siendo el fallback seguro.
- Los archivos originales no deben alterarse durante inferencia.
- La lógica de dominio debe seguir separada del I/O.

## Mapa rápido del código

- [pipeline.py](pipeline.py) concentra configuración, clasificación, preprocesado y orquestación.
- [threading_workers.py](threading_workers.py) contiene las primitivas de concurrencia y contratos thread-safe.
- [main.py](main.py) expone el CLI.
- [tests/](tests/) protege el contrato funcional y de concurrencia.

## Criterio de mantenimiento

Cuando cambie un default, un umbral o un comportamiento de workers, actualiza primero el código y luego [docs/reference/configuration.md](docs/reference/configuration.md). Después ajusta README y este índice si hace falta.

## Estado del documento

Este archivo es intencionalmente corto. Si necesitas una explicación más profunda, enlaza al documento especializado correspondiente en lugar de expandir CLAUDE.md.
