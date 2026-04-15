---
name: feature-debug-config-first
description: Router skill que divide feature development y debugging en skills unitarias para este pipeline.
origin: Local
---

# Feature Debug Config First (Router)

Skill indice para activar skills unitarias. No contiene reglas extensas por si misma.

## Use This Skill To Route

- Config contract and defaults: ../config-contract-pipeline/SKILL.md
- Layered feature implementation: ../feature-layer-separation/SKILL.md
- AI behavior through config: ../ia-behavior-config/SKILL.md
- Debug sequence and observability: ../debugging-playbook-pipeline/SKILL.md
- Regression tests for config changes: ../testing-config-regression/SKILL.md

## Activation Rule

Si la tarea mezcla mas de un dominio, activar primero esta skill y luego las unitarias necesarias.
