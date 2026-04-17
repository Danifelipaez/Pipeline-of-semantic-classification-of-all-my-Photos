#!/usr/bin/env python3
"""Quick sanity check for the optimized pipeline."""

from pathlib import Path
from pipeline import load_config

# Verify config loads with optimized values
cfg = load_config(Path('config.yaml'))
max_side = cfg["preprocessing"]["max_side"]
quality = cfg["preprocessing"]["jpeg_quality"]

print(f"✅ Config loaded successfully")
print(f"   max_side: {max_side} (expected 768)")
print(f"   jpeg_quality: {quality} (expected 80)")

assert max_side == 768, f"max_side should be 768, got {max_side}"
assert quality == 80, f"jpeg_quality should be 80, got {quality}"

print(f"\n✅ All checks passed! Pipeline ready for deployment.")
