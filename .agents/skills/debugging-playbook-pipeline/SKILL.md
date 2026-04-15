---
name: debugging-playbook-pipeline
description: Step-by-step debugging playbook for preprocessing, AI classification, fallback decisions and file operations.
origin: Local
---

# Debugging Playbook Pipeline

Single-responsibility skill for reproducible debugging in this project.

## When to Activate

- Classification results look wrong.
- Too many uncategorized outputs.
- Files are copied/moved to wrong destination.
- Runtime errors in preprocessing or API calls.

## Debug Sequence

1. Run with dry-run and a tiny dataset.
2. Print effective loaded config.
3. Validate image preprocessing (mode, resize, jpeg quality).
4. Trace API request parameters (without leaking sensitive data).
5. Log raw model response and parsed result.
6. Confirm confidence threshold and fallback decision.
7. Verify destination path and operation mode.

## Logging Minimum per Image

- file name
- original size
- payload size
- parsed category
- raw model response
- fallback reason
- final destination

## Done Criteria

- Reproduced issue with deterministic steps.
- Root cause mapped to one layer.
- Fix covered by at least one test.
