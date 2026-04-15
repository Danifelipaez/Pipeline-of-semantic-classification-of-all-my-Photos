---
name: ia-behavior-config
description: Change AI behavior through configuration only: prompt, model, timeout, confidence threshold and fallback category.
origin: Local
---

# IA Behavior Config

Single-responsibility skill for AI behavior tuning without editing core logic.

## When to Activate

- Change model or endpoint.
- Tune prompt strategy.
- Adjust confidence threshold.
- Modify fallback behavior.

## Config-First Parameters

- ollama.url
- ollama.model
- ollama.timeout_seconds
- classification_prompt
- classification.min_confidence
- classification.fallback_category

## Rules

1. Never hardcode model, URL or prompt in classifier logic.
2. Parse model output with stable contract (category|confidence).
3. If confidence < min_confidence, use fallback_category.
4. Keep parser robust to noisy text.

## Quick Validation

- Dry run with a small sample.
- Inspect raw model output.
- Validate category normalization.
- Assert fallback path in tests.
