#!/usr/bin/env python3
"""RoPE position-indexing probe — P10.1.

Measures whether a model can retrieve arr[k] from a 4-element list at varying
context lengths.  Isolates RoPE position-encoding degradation from semantic
difficulty by using single-digit integer values (0-3) with k ∈ {0,1,2,3}.

Baseline accuracy = 0.25 (random chance, 4 classes).

Usage (one cell in the 5×4 sweep):
    python3 rope_position_probe.py \\
        --host 127.0.0.1 --port 8080 \\
        --context-length 8192 \\
        --n-samples 100 --seed 42 \\
        --out /mnt/raid0/llm/epyc-inference-research/benchmarks/results/rope_probe/cell.json

Dry-run (no server required):
    python3 rope_position_probe.py --dry-run --context-length 4096
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Filler text: semantically neutral, no digits that could leak array values.
# A single paragraph is ~100 chars ≈ 25 tokens; repeated to reach target len.
_FILLER_PARA = (
    "The conference proceedings were reviewed by the committee. "
    "All submitted abstracts underwent double-blind evaluation before acceptance. "
    "Authors were notified of decisions within the specified review window. "
    "The schedule was published on the official website for registered participants. "
)

# Conservative chars-per-token estimate (matches context_generator.py convention).
_CHARS_PER_TOKEN = 4

# How many tokens to reserve for the array definition + question + model answer.
# Generous margin so the probe text always fits inside the context window.
_PROMPT_RESERVE_TOKENS = 200


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_filler(target_tokens: int) -> str:
    """Return neutral filler text sized to approximately target_tokens tokens."""
    target_chars = target_tokens * _CHARS_PER_TOKEN
    repeats = max(1, target_chars // len(_FILLER_PARA) + 1)
    blob = (_FILLER_PARA * repeats)[:target_chars]
    return blob


def build_prompt(arr: list[int], k: int, context_length: int) -> str:
    """Build the full probe prompt.

    Layout
    ------
    [filler …]
    --- TASK ---
    arr = [v0, v1, v2, v3]
    What is arr[k]? Answer with a single digit only.

    The array + question are placed at the END of the context so that the
    model must use positional information from a token index that may be
    far from the beginning of the KV cache — the stress point for RoPE.
    """
    filler_tokens = max(0, context_length - _PROMPT_RESERVE_TOKENS)
    filler = _build_filler(filler_tokens)

    task = (
        "\n\n--- TASK ---\n"
        f"arr = [{arr[0]}, {arr[1]}, {arr[2]}, {arr[3]}]\n"
        f"What is arr[{k}]? Answer with a single digit only.\n"
        "Answer:"
    )
    return filler + task


# ---------------------------------------------------------------------------
# Server call
# ---------------------------------------------------------------------------

def _call_completion(
    prompt: str,
    host: str,
    port: int,
    endpoint: str,
    timeout: int = 120,
) -> str | None:
    """POST to llama-server /completion and return the generated text, or None on error."""
    url = f"http://{host}:{port}{endpoint}"
    payload = {
        "prompt": prompt,
        "n_predict": 4,        # single digit + optional whitespace
        "temperature": 0.0,    # greedy — deterministic
        "cache_prompt": False,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("content", "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] request failed: {exc}", file=sys.stderr)
        return None


def _parse_answer(text: str | None) -> int | None:
    """Extract the first digit 0-3 from model output, or None if unparseable."""
    if text is None:
        return None
    for ch in text:
        if ch in "0123":
            return int(ch)
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_probe(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    records: list[dict] = []
    correct = 0
    skipped = 0

    for i in range(args.n_samples):
        # Sample a random array and a random index
        arr = [rng.randint(0, 3) for _ in range(4)]
        k = rng.randint(0, 3)
        expected = arr[k]

        prompt = build_prompt(arr, k, args.context_length)

        if args.dry_run:
            # Print one sample and exit immediately — no server call
            print(f"=== DRY RUN — sample {i} ===")
            print(f"arr={arr}, k={k}, expected={expected}")
            print(f"prompt length (chars): {len(prompt)}")
            print(f"prompt length (est. tokens): {len(prompt) // _CHARS_PER_TOKEN}")
            print("--- prompt tail (last 400 chars) ---")
            print(prompt[-400:])
            return

        raw = _call_completion(
            prompt,
            host=args.host,
            port=args.port,
            endpoint=args.endpoint,
        )
        predicted = _parse_answer(raw)

        hit = (predicted == expected) if predicted is not None else False
        if predicted is None:
            skipped += 1
        if hit:
            correct += 1

        records.append({
            "sample_idx": i,
            "arr": arr,
            "k": k,
            "expected": expected,
            "raw_output": raw,
            "predicted": predicted,
            "correct": hit,
        })

        # Incremental progress
        total_answered = i + 1 - skipped
        running_acc = correct / total_answered if total_answered else 0.0
        print(
            f"[{i+1:3d}/{args.n_samples}] k={k} expected={expected} "
            f"got={predicted!r}  running_acc={running_acc:.3f}",
            flush=True,
        )

    total_answered = args.n_samples - skipped
    accuracy = correct / total_answered if total_answered else 0.0

    result = {
        "context_length": args.context_length,
        "n_samples": args.n_samples,
        "n_answered": total_answered,
        "n_skipped": skipped,
        "n_correct": correct,
        "accuracy": accuracy,
        "baseline_chance": 0.25,
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "endpoint": args.endpoint,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"\nDone. accuracy={accuracy:.3f} (baseline=0.25)  "
        f"skipped={skipped}  saved → {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoPE position-indexing probe (P10.1). "
                    "Runs 100 arr[k] lookups at a given context length against a local llama-server.",
    )
    p.add_argument("--host", default="127.0.0.1", help="llama-server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="llama-server port (default: 8080)")
    p.add_argument(
        "--endpoint", default="/completion",
        help="Server endpoint (default: /completion)",
    )
    p.add_argument(
        "--context-length", type=int, required=True,
        help="Target context length in tokens (e.g. 4096, 8192, 16384, 32768)",
    )
    p.add_argument("--n-samples", type=int, default=100, help="Samples per cell (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument(
        "--out",
        default="/mnt/raid0/llm/epyc-inference-research/benchmarks/results/rope_probe/results.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build and print one sample prompt without calling the server, then exit",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_probe(_parse_args())
