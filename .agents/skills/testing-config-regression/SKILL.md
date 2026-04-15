---
name: testing-config-regression
description: Add focused tests for config-driven behavior and regression safety in the semantic photo pipeline.
origin: Local
---

# Testing Config Regression

Single-responsibility skill for tests that protect config-driven behavior.

## When to Activate

- Add a new config key.
- Change category extraction or fallback logic.
- Refactor pipeline orchestration.

## Mandatory Test Cases

1. Loads defaults when key is missing.
2. Applies config override correctly.
3. Rejects invalid operation value.
4. Rejects empty categories.
5. Parses category|confidence correctly.
6. Sends low-confidence outputs to fallback category.
7. Dry-run does not move/copy files.
8. Classification errors continue processing and log error.

## Test Design Rules

- Mock external HTTP calls.
- Use tmp_path for filesystem isolation.
- Assert behavior, not internal implementation details.
- Keep one behavior assertion focus per test.
