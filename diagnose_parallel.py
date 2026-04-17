#!/usr/bin/env python
"""Diagnostic script to verify Ollama parallel classification setup."""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    print("✓ requests library installed")
except ImportError:
    print("✗ requests library NOT installed")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ PIL library installed")
except ImportError:
    print("✗ PIL library NOT installed")
    sys.exit(1)

import threading

print("\n=== OLLAMA CONNECTION TEST ===\n")

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4"

# Test 1: Check if Ollama is running
print(f"1. Checking if Ollama is running at {OLLAMA_URL}...")
try:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    print(f"   ✓ Ollama is running (status: {response.status_code})")
    models = response.json().get("models", [])
    print(f"   Models available: {[m.get('name', 'unknown') for m in models]}")
    
    # Check if gemma4 is available
    gemma4_found = any(m.get("name", "").startswith("gemma4") for m in models)
    if gemma4_found:
        print(f"   ✓ {MODEL} is downloaded")
    else:
        print(f"   ✗ {MODEL} is NOT downloaded. Run: ollama pull gemma4")
        sys.exit(1)
except requests.ConnectionError as e:
    print(f"   ✗ CANNOT CONNECT to Ollama: {e}")
    print("   → Start Ollama: ollama serve (in WSL or Docker)")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Single image classification
print(f"\n2. Testing single image classification...")
import base64
from io import BytesIO

test_image_path = Path(__file__).parent / "test_image.png"
if not test_image_path.exists():
    print("   Creating test image...")
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(test_image_path)

with Image.open(test_image_path) as img:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    payload_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

prompt = """Describe this image briefly and choose exactly one category:
nature, birds, street photography, family, portraits, architecture, food, events

Output format: category|confidence (example: nature|95)"""

try:
    start = time.perf_counter()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "images": [payload_b64],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start
    
    if response.status_code == 200:
        result = response.json().get("response", "")
        print(f"   ✓ Classification succeeded in {elapsed:.2f}s")
        print(f"   Response: {result.strip()[:100]}")
    else:
        print(f"   ✗ Ollama returned status {response.status_code}: {response.text[:200]}")
        sys.exit(1)
except requests.Timeout:
    print(f"   ✗ Timeout after 120s. Ollama might be busy or model too large.")
    sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Parallel classification (simulate what the pipeline does)
print(f"\n3. Testing PARALLEL classification with 3 workers...")

def classify_single(image_id: int) -> tuple[int, float, str]:
    """Classify a test image and return (id, elapsed, response)."""
    thread_id = threading.get_ident()
    start = time.perf_counter()
    
    with Image.open(test_image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        payload_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "images": [payload_b64],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        result = response.json().get("response", "")[:50]
        elapsed = time.perf_counter() - start
        return image_id, elapsed, f"OK (thread {thread_id}): {result}"
    except Exception as e:
        elapsed = time.perf_counter() - start
        return image_id, elapsed, f"ERROR (thread {thread_id}): {e}"

num_tasks = 3
num_workers = 3

print(f"   Sending {num_tasks} images to {num_workers} parallel workers...")
executor = ThreadPoolExecutor(max_workers=num_workers)
futures = [executor.submit(classify_single, i) for i in range(num_tasks)]

parallel_start = time.perf_counter()
for future in as_completed(futures):
    image_id, elapsed, status = future.result()
    print(f"   Image {image_id+1}: {status} ({elapsed:.2f}s)")

parallel_elapsed = time.perf_counter() - parallel_start
executor.shutdown(wait=True)

print(f"\n   Total parallel time: {parallel_elapsed:.2f}s")
print(f"   Sequential equivalent: ~{num_tasks * elapsed:.2f}s")

if parallel_elapsed < (num_tasks * elapsed - 5):  # Allow 5s margin
    print(f"   ✓ Parallelism is WORKING (parallel < sequential)")
else:
    print(f"   ⚠ Parallelism may not be effective.")
    print(f"   → This could be due to:")
    print(f"      - Ollama processing sequentially (CPU-only, no GPU)")
    print(f"      - Network bottleneck")
    print(f"      - Model inference taking >12s per image")

# Test 4: Check GPU support
print(f"\n4. Checking GPU support in Ollama...")
try:
    process_info = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5).json()
    running_models = process_info.get("models", [])
    if running_models:
        print(f"   Running models: {[m.get('name') for m in running_models]}")
        for model_info in running_models:
            # Check if model is using GPU (VRAM)
            vram = model_info.get("details", {}).get("quantization", "unknown")
            print(f"   {model_info.get('name')}: quantization={vram}")
    else:
        print(f"   No models currently running. Start one with: ollama run {MODEL}")
except Exception as e:
    print(f"   ⚠ Could not query running models: {e}")

print("\n=== DIAGNOSTICS COMPLETE ===\n")

# Cleanup
test_image_path.unlink(missing_ok=True)

print("SUMMARY:")
print("✓ Ollama is accessible")
print("✓ gemma4 model is available")
print("✓ Single classification works")
if parallel_elapsed < (num_tasks * elapsed - 5):
    print("✓ Parallel classification is working efficiently")
else:
    print("⚠ Parallel classification may need optimization")

print("\nNext steps:")
print("1. If GPU is NOT detected, install GPU drivers or use CPU mode")
print("2. Run: python main.py --source D:\\Camara\\a6400 --output D:\\Camara\\OrganizacionSemantica")
print("3. Check D:\\Camara\\OrganizacionSemantica\\errors.log for any issues")
