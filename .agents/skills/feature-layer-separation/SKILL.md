---
name: feature-layer-separation
description: Implement new features by separating CLI, application flow, domain rules and infrastructure concerns.
origin: Local
---

# Feature Layer Separation

Single-responsibility skill for architecture boundaries when developing features.

## When to Activate

- Add a new feature in pipeline flow.
- Refactor large functions into smaller units.
- Reduce coupling between business logic and IO.

## Layer Boundaries

- Entry layer: CLI parsing and config loading.
- Application layer: orchestration of processing steps.
- Domain layer: category extraction, confidence rules, fallback decisions.
- Infrastructure layer: filesystem, HTTP client, logging.

## Rules

1. Keep one reason to change per function.
2. Pass dependencies as arguments.
3. Domain functions should be pure when possible.
4. Infrastructure calls should not contain business decisions.

## Refactor Flow

1. Identify mixed-responsibility function.
2. Extract domain decisions first.
3. Wrap IO in dedicated helpers/adapters.
4. Wire in application orchestrator.
5. Cover with tests before and after.
