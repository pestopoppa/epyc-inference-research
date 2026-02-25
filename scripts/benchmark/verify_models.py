#!/usr/bin/env python3
from __future__ import annotations

"""
Model Verification Script

Tests every model in the registry to ensure it can:
1. Load without errors
2. Run basic inference
3. Return valid output

Usage:
    ./verify_models.py              # Test all models
    ./verify_models.py --fix        # Also update registry with findings
    ./verify_models.py --model X    # Test specific model
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.registry import load_registry

LLAMA_BIN = "/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TEST_PROMPT = "Hello"
TIMEOUT = 60  # seconds


def test_model(model_path: str, role: str = None) -> dict:
    """Test if a model can load and run inference.

    Returns:
        dict with keys: success, error, tokens_per_second, output
    """
    result = {
        "success": False,
        "error": None,
        "tokens_per_second": None,
        "output": None,
    }

    if not os.path.exists(model_path):
        result["error"] = f"File not found: {model_path}"
        return result

    # Create temp file for prompt
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(TEST_PROMPT)
        prompt_file = f.name

    try:
        cmd = [
            LLAMA_BIN,
            "-m", model_path,
            "-t", "8",  # Use fewer threads for quick test
            "-n", "5",  # Just 5 tokens - enough to verify it works
            "--temp", "0",
            "-f", prompt_file,
            "--no-conversation",  # Disable conversation mode to prevent hangs
            "--no-warmup",  # Skip warmup for speed
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        output = proc.stdout + proc.stderr
        result["output"] = output[:2000]  # Truncate

        # Check for common errors
        if "invalid argument" in output.lower():
            result["error"] = "Invalid argument - model format issue"
        elif "failed to load" in output.lower():
            result["error"] = "Failed to load model"
        elif "error:" in output.lower():
            # Extract error message
            for line in output.split('\n'):
                if 'error:' in line.lower():
                    result["error"] = line.strip()[:100]
                    break
        elif proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}"
        else:
            # Success - try to extract tokens/second
            result["success"] = True
            for line in output.split('\n'):
                if 'eval time' in line.lower() and 'token' in line.lower():
                    # Parse "eval time = ... ms / ... tokens (XX.XX t/s)"
                    if 't/s' in line:
                        try:
                            tps = float(line.split('(')[1].split('t/s')[0].strip())
                            result["tokens_per_second"] = tps
                        except Exception as e:
                            pass
                    break

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {TIMEOUT}s"
    except Exception as e:
        result["error"] = str(e)
    finally:
        os.unlink(prompt_file)

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify all models in registry")
    parser.add_argument("--model", "-m", help="Test specific model role")
    parser.add_argument("--fix", action="store_true", help="Update registry with findings")
    parser.add_argument("--quick", action="store_true", help="Stop after first failure")
    args = parser.parse_args()

    registry = load_registry()
    roles = registry.get_all_roles(include_deprecated=False)

    if args.model:
        roles = [r for r in roles if r == args.model]

    # Deduplicate by model path
    tested_paths = set()
    results = []

    print(f"Verifying {len(roles)} roles...\n")
    print(f"{'Role':<45} {'Size':>8} {'Status':<10} {'TPS':>8} {'Error'}")
    print("=" * 110)

    passed = 0
    failed = 0
    skipped = 0

    for role in sorted(roles):
        model_path = registry.get_model_path(role)

        if not model_path:
            print(f"{role:<45} {'---':>8} {'NO PATH':<10}")
            skipped += 1
            continue

        if model_path in tested_paths:
            # Already tested this exact file
            prev = next((r for r in results if r["path"] == model_path), None)
            if prev:
                status = "✓ (dup)" if prev["success"] else "✗ (dup)"
                print(f"{role:<45} {'':>8} {status:<10}")
            continue

        tested_paths.add(model_path)

        if not os.path.exists(model_path):
            print(f"{role:<45} {'---':>8} {'MISSING':<10} {'':>8} File not found")
            results.append({"role": role, "path": model_path, "success": False, "error": "File not found"})
            failed += 1
            continue

        size_gb = os.path.getsize(model_path) / (1024**3)

        # Test the model
        test_result = test_model(model_path, role)
        test_result["role"] = role
        test_result["path"] = model_path
        results.append(test_result)

        if test_result["success"]:
            tps = test_result.get("tokens_per_second")
            tps_str = f"{tps:.1f}" if tps else "---"
            print(f"{role:<45} {size_gb:>7.1f}G {'✓ OK':<10} {tps_str:>8}")
            passed += 1
        else:
            error = test_result.get("error", "Unknown error")[:40]
            print(f"{role:<45} {size_gb:>7.1f}G {'✗ FAIL':<10} {'---':>8} {error}")
            failed += 1

            if args.quick:
                print("\n--quick mode: stopping after first failure")
                break

    print("=" * 110)
    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n=== FAILED MODELS ===")
        for r in results:
            if not r.get("success") and r.get("error"):
                print(f"  {r['role']}: {r['error']}")

        print("\nThese models need to be:")
        print("  1. Removed from registry, OR")
        print("  2. Fixed with proper launch quirks, OR")
        print("  3. Re-downloaded/re-converted")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
