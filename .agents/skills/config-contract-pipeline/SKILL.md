---
name: config-contract-pipeline
description: Define, validate and evolve config.yaml contract for this photo classification pipeline without breaking existing behavior.
origin: Local
---

# Config Contract Pipeline

Single-responsibility skill for configuration schema and defaults.

## When to Activate

- Add or change keys in config.yaml.
- Introduce new behavior flags.
- Keep backward compatibility of configuration.

## Scope

- Define stable config contract.
- Keep defaults centralized in one place.
- Validate required and optional fields.

## Required Rules

1. No hardcoded behavior in domain logic.
2. Every new config key must have a default.
3. Validate operation in allowed set (copy, move).
4. Validate categories is non-empty.
5. Keep type casting explicit at boundary.

## Contract Template

```yaml
source: ./photos
output: ./sorted
operation: copy
categories:
  - nature
  - birds

ollama:
  url: http://localhost:11434
  model: gemma4
  timeout_seconds: 120

classification:
  min_confidence: 0.85
  fallback_category: uncategorized

classification_prompt: |
  Choose one category from: {categories}
  Respond with category|confidence

preprocessing:
  max_side: 1024
  jpeg_quality: 85
```

## Implementation Checklist

- Add key to defaults dictionary.
- Merge loaded config over defaults.
- Validate value ranges and enums.
- Add tests for default + override + invalid value.
