#!/usr/bin/env python3
from __future__ import annotations

"""TTFT benchmark comparing JSON vs TOON encoding for orchestrator."""
import json
import time
import httpx
from typing import Callable

# Test data: typical orchestrator outputs
TEST_CASES = {
    "file_listing": {
        "files": [
            {"name": "main.py", "type": "file", "size": 1234, "modified": "2026-01-28"},
            {"name": "utils.py", "type": "file", "size": 567, "modified": "2026-01-27"},
            {"name": "tests", "type": "dir", "size": 0, "modified": "2026-01-26"},
            {"name": "config.yaml", "type": "file", "size": 234, "modified": "2026-01-25"},
            {"name": "README.md", "type": "file", "size": 1500, "modified": "2026-01-24"},
            {"name": "requirements.txt", "type": "file", "size": 89, "modified": "2026-01-23"},
            {"name": "__pycache__", "type": "dir", "size": 0, "modified": "2026-01-22"},
            {"name": "output.log", "type": "file", "size": 45678, "modified": "2026-01-21"},
        ]
    },
    "grep_results": {
        "query": "error",
        "hits": [
            {"file": "main.py", "line": 42, "content": "    raise ValueError('Critical error')"},
            {"file": "utils.py", "line": 78, "content": "    logger.error('Failed to process')"},
            {"file": "tests/test_main.py", "line": 15, "content": "    with pytest.raises(RuntimeError):"},
            {"file": "config.yaml", "line": 23, "content": "error_handling: strict"},
            {"file": "output.log", "line": 156, "content": "[ERROR] Connection timed out"},
            {"file": "output.log", "line": 287, "content": "[ERROR] Database unavailable"},
        ]
    },
    "escalation_context": {
        "failure_count": 2,
        "error_category": "FORMAT",
        "gate_name": "schema",
        "previous_attempts": [
            {"role": "coder", "error": "JSON parse failed at line 42", "tokens": 1234, "latency_ms": 456},
            {"role": "coder", "error": "Missing required field 'id'", "tokens": 1456, "latency_ms": 523},
        ]
    }
}


def encode_json(data: dict) -> str:
    return json.dumps(data, indent=2)


def encode_toon(data: dict) -> str:
    """Simple TOON encoder for benchmark purposes."""
    # Simplified TOON encoding for arrays of objects
    def encode_array_of_objects(name: str, items: list) -> str:
        if not items:
            return f"{name}[]"

        fields = list(items[0].keys())
        header = f"{name}[{len(items)}]{{{','.join(fields)}}}:"
        rows = []
        for item in items:
            row = ",".join(str(item.get(f, "")) for f in fields)
            rows.append(f"  {row}")
        return header + "\n" + "\n".join(rows)

    parts = []
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            parts.append(encode_array_of_objects(key, value))
        elif isinstance(value, dict):
            # Simple nested object
            inner = ",".join(f"{k}={v}" for k, v in value.items())
            parts.append(f"{key}:{{{inner}}}")
        else:
            parts.append(f"{key}={value}")
    return "\n".join(parts)


def measure_ttft(server_port: int, prompt: str, n_runs: int = 3) -> dict:
    """Measure time-to-first-token for a prompt."""
    url = f"http://127.0.0.1:{server_port}/completion"

    ttfts = []
    for _ in range(n_runs):
        payload = {
            "prompt": prompt,
            "n_predict": 50,  # Just need first token timing
            "temperature": 0.1,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, json=payload)
                data = resp.json()
                # TTFT is prompt_eval_time (prefill)
                ttft = data.get("timings", {}).get("prompt_ms", 0) / 1000
                ttfts.append(ttft)
        except Exception as e:
            print(f"Error: {e}")
            continue

    if not ttfts:
        return {"error": "no successful runs"}

    return {
        "avg_ttft_s": sum(ttfts) / len(ttfts),
        "min_ttft_s": min(ttfts),
        "max_ttft_s": max(ttfts),
        "runs": len(ttfts),
    }


def count_tokens_approx(text: str) -> int:
    """Rough token count (avg 4 chars/token for English)."""
    return len(text) // 4


def main():
    print("=" * 70)
    print("TTFT Benchmark: JSON vs TOON Encoding")
    print("=" * 70)

    SERVER_PORT = 8080  # frontdoor

    # Check server health
    try:
        resp = httpx.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=5)
        if resp.json().get("status") != "ok":
            print(f"Server not healthy!")
            return
    except Exception as e:
        print(f"Server not reachable: {e}")
        return

    print(f"Using server: 127.0.0.1:{SERVER_PORT}\n")

    results = []

    for case_name, data in TEST_CASES.items():
        print(f"\n{'='*50}")
        print(f"Test Case: {case_name}")
        print("=" * 50)

        # Encode both ways
        json_str = encode_json(data)
        toon_str = encode_toon(data)

        json_tokens = count_tokens_approx(json_str)
        toon_tokens = count_tokens_approx(toon_str)
        reduction = (1 - toon_tokens / json_tokens) * 100 if json_tokens > 0 else 0

        print(f"JSON: {len(json_str)} chars (~{json_tokens} tokens)")
        print(f"TOON: {len(toon_str)} chars (~{toon_tokens} tokens)")
        print(f"Reduction: {reduction:.1f}%")

        # Build test prompts
        base_prompt = "Given the following data, summarize the key findings:\n\n"
        json_prompt = base_prompt + json_str + "\n\nSummary:"
        toon_prompt = base_prompt + toon_str + "\n\nSummary:"

        print(f"\nMeasuring TTFT (3 runs each)...")

        # Measure TTFT
        json_ttft = measure_ttft(SERVER_PORT, json_prompt)
        toon_ttft = measure_ttft(SERVER_PORT, toon_prompt)

        print(f"JSON TTFT: {json_ttft.get('avg_ttft_s', 0):.3f}s")
        print(f"TOON TTFT: {toon_ttft.get('avg_ttft_s', 0):.3f}s")

        if json_ttft.get('avg_ttft_s') and toon_ttft.get('avg_ttft_s'):
            ttft_reduction = json_ttft['avg_ttft_s'] - toon_ttft['avg_ttft_s']
            ttft_pct = (ttft_reduction / json_ttft['avg_ttft_s']) * 100
            print(f"TTFT Reduction: {ttft_reduction:.3f}s ({ttft_pct:.1f}%)")

        results.append({
            "case": case_name,
            "json_chars": len(json_str),
            "toon_chars": len(toon_str),
            "json_tokens": json_tokens,
            "toon_tokens": toon_tokens,
            "token_reduction_pct": reduction,
            "json_ttft_s": json_ttft.get('avg_ttft_s', 0),
            "toon_ttft_s": toon_ttft.get('avg_ttft_s', 0),
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Case':<25} {'Token Δ':<10} {'JSON TTFT':<12} {'TOON TTFT':<12} {'TTFT Δ':<10}")
    print("-" * 70)

    for r in results:
        ttft_delta = r['json_ttft_s'] - r['toon_ttft_s']
        print(f"{r['case']:<25} {r['token_reduction_pct']:>6.1f}%   {r['json_ttft_s']:>8.3f}s    {r['toon_ttft_s']:>8.3f}s    {ttft_delta:>+.3f}s")

    # Calculate averages
    avg_token_reduction = sum(r['token_reduction_pct'] for r in results) / len(results)
    avg_json_ttft = sum(r['json_ttft_s'] for r in results) / len(results)
    avg_toon_ttft = sum(r['toon_ttft_s'] for r in results) / len(results)
    avg_ttft_reduction = avg_json_ttft - avg_toon_ttft

    print("-" * 70)
    print(f"{'AVERAGE':<25} {avg_token_reduction:>6.1f}%   {avg_json_ttft:>8.3f}s    {avg_toon_ttft:>8.3f}s    {avg_ttft_reduction:>+.3f}s")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if avg_ttft_reduction > 0:
        print(f"✅ TOON is FASTER by {avg_ttft_reduction:.3f}s average ({avg_ttft_reduction/avg_json_ttft*100:.1f}% improvement)")
        print(f"   Combined with {avg_token_reduction:.1f}% token reduction = RECOMMENDED")
    else:
        print(f"⚠️ TOON is SLOWER by {-avg_ttft_reduction:.3f}s average")
        print(f"   Token reduction of {avg_token_reduction:.1f}% may not compensate")

    # Save results
    with open("/tmp/ttft_toon_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/ttft_toon_results.json")


if __name__ == "__main__":
    main()
